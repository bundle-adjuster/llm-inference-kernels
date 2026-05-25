# Foundations — a book for this repo

> A first-principles companion to the kernels and journey docs. Built so
> that after reading + drilling + walkthrough, you can re-implement the
> Phase 1 and Phase 2 work from memory and teach it cold.
>
> Five parts:
> 1. **What we're actually computing** (this part)
> 2. The GPU mental model
> 3. Decode attention, naive → fast
> 4. KV-cache compression
> 5. Cross-cutting lessons
>
> Each chapter has **checkpoint questions** inline. Don't continue past
> a checkpoint until you can answer it in your own words — the rest of
> the book builds on those answers.

---

# Part 1 — What we're actually computing

This part has nothing to do with CUDA. It's about the operation we want
the GPU to do. Get this right first, or the rest of the book is just
syntax.

## 1.1 The big picture — where attention sits

A Large Language Model like Llama 3.1 8B is a tower of *transformer
blocks*, plus an embedding at the bottom and a classifier at the top.
Llama 3 8B has 32 blocks stacked on top of each other. Each block does
two things to its input:

```
input  →  attention  →  residual add  →  MLP  →  residual add  →  output
                        + LayerNorm           + LayerNorm
```

For our purposes the only thing inside a block that matters is the
**attention** layer. Each block has its own attention; they don't share
weights. When the model "thinks" about token 47 of the input, that
token's representation flows up through 32 attention computations
before becoming a prediction for token 48.

The MLP and the LayerNorms are easy from a kernel standpoint — they're
elementwise + matmul, fast on standard libraries. The two things that
dominate inference cost are:

1. **Attention** in every layer. This is what makes long-context inference
   slow: it has to look at every prior token.
2. **The big matmuls** in the MLP and the QKV/output projections. With
   the model in fp16, these are weight-bandwidth-bound; with 4-bit
   weight quantization they get much faster.

This repo attacks both: **Phase 1 + 2 = attention + its KV cache**,
**Phase 3 = quantized matmul**. Phase 4 is the integration that ties
them together.

Why is attention hard? Because of two things:
- It's **all-to-all** — every output token depends on every prior token.
  So work grows with sequence length.
- Its memory cost is the *KV cache*, which can dominate VRAM at long
  context.

Phase 1 makes the attention compute fast. Phase 2 shrinks the KV cache.

> **Checkpoint 1.1**
> - Why do we care more about kernel speed for attention and matmul than
>   for LayerNorm?
> - In a 32-block model, how many distinct attention computations happen
>   per decoded token?

---

## 1.2 Attention from scratch — the mechanical view

Forget Q, K, V for a moment. Here is what attention *does*, in plain
English:

> Given a sequence of tokens, for each token, compute a new
> representation that is a **weighted average of the representations of
> all the other tokens**, where the weights say "how relevant is this
> other token to me?"

That's it. The whole purpose of attention is to mix information across
positions, weighted by relevance. Everything else is plumbing for that
idea.

Now the plumbing.

Each token in the input has a `d`-dimensional representation. Call this
representation `x`. We're going to compute three different "views" of
each token:

- **Query** (`q`): "what am I looking for in the other tokens?"
- **Key** (`k`): "what do I represent, that other tokens might look for?"
- **Value** (`v`): "what information do I carry to share?"

These are three linear projections of the same input `x`:

```
q = W_q · x       (size d)
k = W_k · x       (size d)
v = W_v · x       (size d)
```

`W_q`, `W_k`, `W_v` are learned weight matrices. For each token we get
a Q, a K, and a V. Stacked across the whole sequence:

- `Q` is shape `[seq_len, d]`
- `K` is shape `[seq_len, d]`
- `V` is shape `[seq_len, d]`

Now the actual attention computation. For token `i`'s output, we want a
weighted average of *all* tokens' V vectors. The weights come from
matching token `i`'s Q against every token's K:

```
score_i_j  =  q_i · k_j                    (one scalar per pair)
weight_i_j =  softmax(score_i_j over j)    (weights sum to 1 over j)
output_i   =  Σ_j  weight_i_j · v_j        (weighted average of V's)
```

The softmax turns scores into a probability distribution. Tokens with
high scores get most of the weight; low-score tokens contribute almost
nothing.

In matrix form across the whole sequence:

```
S  =  Q · K^T           shape [seq_len, seq_len]   (every-pair scores)
P  =  softmax(S, dim=last)                          (rows sum to 1)
O  =  P · V                                          (output, [seq_len, d])
```

Two more details that matter:

**The 1/√d scaling.** The dot products `q · k` have variance proportional
to `d`. For large `d`, scores get huge, and softmax saturates (one weight
goes to 1, the rest go to 0 — useless). The fix: divide by `√d` before
softmax, so the variance stays roughly 1 regardless of `d`. This is the
`softmax_scale = 1 / sqrt(head_dim)` you see everywhere.

**Causal masking.** When generating text, token `i` is only allowed to
look at tokens `≤ i`. We can't let it cheat by looking at future tokens
during training. The fix: set `S_i_j = −∞` for `j > i`, so `softmax` gives
those positions weight 0. We'll come back to this.

### A worked numerical example

Let's pick tiny numbers: `d = 4`, `seq_len = 3`. We have three tokens.
Their Q, K, V (just made up):

```
Q = [[ 1,  0,  0,  0],     (token 0's query)
     [ 0,  1,  0,  0],     (token 1's)
     [ 0,  0,  1,  0]]     (token 2's)

K = [[ 1,  0,  0,  0],
     [ 0,  1,  0,  0],
     [ 0,  0,  1,  0]]

V = [[10, 20, 30, 40],     (token 0's value)
     [50, 60, 70, 80],
     [90,100,110,120]]
```

Compute `S = Q · K^T`:

```
S[0,0] = 1·1 + 0 + 0 + 0 = 1.0
S[0,1] = 1·0 + 0·1 + 0 + 0 = 0
S[0,2] = 0
... and so on
S  =  [[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]]
```

Apply softmax (over each row), with `d=4` so scale = `1/√4 = 0.5`. The
scaled scores are `S/2 = diag(0.5)` and off-diagonal 0. Row 0's softmax:

```
softmax([0.5, 0, 0]) ≈ [0.51, 0.245, 0.245]
```

Note that even with a score of 0.5 vs 0, the high-score position only
gets weight 0.51 — softmax with score ranges around 0 to 1 is *not very
peaky*. That's the point of scaling.

Then `O = P · V`:

```
O[0] = 0.51·[10,20,30,40] + 0.245·[50,60,70,80] + 0.245·[90,100,110,120]
     = [5.1, 10.2, 15.3, 20.4]
     + [12.25, 14.7, 17.15, 19.6]
     + [22.05, 24.5, 26.95, 29.4]
     = [39.4, 49.4, 59.4, 69.4]
```

So token 0's new representation is `[39.4, 49.4, 59.4, 69.4]` — a mix
weighted ~half towards itself, ~quarter each towards the others.

The mechanical takeaway: **attention is a soft, weighted average of V's,
indexed by Q-against-K similarity.**

> **Checkpoint 1.2**
> - Without looking, write the attention formula `O = ...` in terms of
>   `Q, K, V, d`.
> - Why do we divide by `√d` before softmax? What goes wrong if we don't?
> - If `Q` and `K` are identical and orthonormal across tokens, what is
>   `softmax(Q·K^T / √d)` — close to identity or close to uniform?

---

## 1.3 Multi-head attention

One pass of attention as described above is *one head*. Real models do
multi-head attention: split the `d`-dimensional space into `n_heads`
chunks of `head_dim = d / n_heads`, run attention independently in each
chunk, then concatenate.

The motivation: each head can learn a different "pattern of attention" —
one might focus on syntactic relations, another on long-distance
co-references, another on local n-grams. Empirically this works much
better than a single fat attention computation.

In Llama 3 8B, the QKV projections produce vectors of dimension
`n_heads × head_dim = 32 × 128 = 4096`. We reshape into `(n_heads,
head_dim) = (32, 128)` and run attention independently in each of the
32 heads. The outputs `[32, head_dim]` get concatenated back to a
`4096`-dim vector and projected to the original `d` via `W_o`.

Tensor shapes during a *training-style* forward pass (where everything
fits in memory):

```
input       x : [batch, seq_len, d]                   d = 4096

q,k,v projections produce:
            q : [batch, seq_len, n_heads, head_dim]   = [b, s, 32, 128]
            k : [batch, seq_len, n_heads, head_dim]
            v : [batch, seq_len, n_heads, head_dim]

transpose to: [batch, n_heads, seq_len, head_dim]    (heads outer for compute)

per-head:
            s = q · k^T / √head_dim   shape [b, n_heads, seq_len, seq_len]
            p = softmax(s + mask)
            o = p · v                  shape [b, n_heads, seq_len, head_dim]

transpose back, reshape, project:
            out = W_o · concat(o per head)            [batch, seq_len, d]
```

The kernels we write live at the *per-head* layer, parallelised across
the `n_heads` dimension. Each thread block handles one (batch, head)
pair. That's the grid you saw in `grid(batch, n_heads)` in v0 → v3.

> **Checkpoint 1.3**
> - For Llama 3 8B with `d=4096, n_heads=32`, what is `head_dim`?
> - In multi-head attention, do the different heads share weights? Share
>   inputs? Share outputs?
> - If we have batch=8, n_heads=32, seqlen=4096, head_dim=128, how big
>   is `Q` (in fp16)?

---

## 1.4 GQA — Grouped Query Attention

In a vanilla transformer, each head has its own Q, K, V projections. So
the K and V tensors have shape `[batch, n_heads, seq_len, head_dim]` —
one per head, just like Q.

For inference, we have to *cache* K and V across decode steps (more on
this in §1.6). That cache scales with `n_heads`. For Llama 3 8B with 32
heads × 32 layers, the cache gets large fast.

GQA cuts the cost: instead of one K, V pair per query head, **multiple
query heads share the same K, V pair**. Llama 3 8B uses `n_heads = 32`
query heads but only `n_kv_heads = 8` KV heads. Each KV head is shared
by `n_heads / n_kv_heads = 4` query heads.

So the shapes diverge:

```
Q : [batch, n_heads,    seq_len, head_dim]   = [b, 32, s, 128]
K : [batch, n_kv_heads, seq_len, head_dim]   = [b,  8, s, 128]   (4× smaller)
V : [batch, n_kv_heads, seq_len, head_dim]   = [b,  8, s, 128]
```

For computation, we logically expand K and V back to 32 heads by
repeating each KV head 4 times (this is `repeat_kv` in HF transformers
source). But for *storage*, K and V keep the smaller `n_kv_heads`
dimension — that's where GQA's memory savings come from.

In our kernels, we don't physically expand. We use an **index map**:
when computing attention for query head `h`, look up K, V from kv head
`kv(h) = h / (n_heads / n_kv_heads)`. For Llama 3 8B that's
`kv(h) = h / 4`. So:

- query heads 0, 1, 2, 3   → kv head 0
- query heads 4, 5, 6, 7   → kv head 1
- ...
- query heads 28, 29, 30, 31 → kv head 7

This is the `kv_head_idx = head_idx / (n_heads / n_kv_heads)` line in
every kernel we wrote.

> **Checkpoint 1.4**
> - In Llama 3 8B, how many bytes does the K tensor take per token (one
>   layer, fp16)? Hint: `n_kv_heads × head_dim × 2`.
> - For query head 17, which kv head does it index into?
> - GQA saves memory; does it cost anything? (Hint: think about model
>   quality, not compute.)

---

## 1.5 Prefill vs decode — the M=1 problem

So far we've described attention as if all `seq_len` tokens are
available at once and processed in parallel. That's how *training*
works, and also how the **prefill** phase of inference works: when the
user gives a prompt of, say, 512 tokens, the model processes all 512 in
one shot.

Then comes **decode**: the model generates new tokens one at a time.
For decode step `t`, the input is *just one new token's representation*.
The model needs to produce a Q for that one new token, attend it
against all *previously seen* K and V (the cache), and output one new
representation.

Decode attention shape:

```
Q : [batch, n_heads,    1,         head_dim]   ← only one position
K : [batch, n_kv_heads, seqlen_kv, head_dim]   ← all prior positions
V : [batch, n_kv_heads, seqlen_kv, head_dim]   ← all prior positions
out : [batch, n_heads,  1,         head_dim]
```

`seqlen_kv` is the current context length — the number of tokens already
processed. It grows by one with every decoded token.

This `Q` shape — a single query position — is what we mean by the
**M = 1 problem**. Tensor Cores on modern GPUs are designed for matrix
multiplies where M, N, K are all reasonably large (typical Tensor Core
shape: M=N=K=16). When M=1, you're doing a `(1 × d) · (d × seq_len)`
matmul. The GPU can do this, but most of the silicon dedicated to
matmul is wasted — you're really doing a *gemv*, a matrix-vector
product, not a matrix-matrix product.

That's the core reason decode attention is harder than prefill:
- Prefill is **compute-bound for long prompts** and maps well to
  FlashAttention-2.
- Decode is **memory-bound** in the limit — read the entire KV cache,
  do tiny per-position FLOPs, write one output. Memory bandwidth, not
  Tensor Core throughput, is the ceiling. (Or so the theory goes — Phase
  1 will show it's more nuanced for our specific workload.)

The split between prefill and decode is why this repo focuses on a
**decode** kernel for Phase 1. Decode dominates chat serving wall-clock
(one token at a time × hundreds of generated tokens). Prefill happens
once per request.

> **Checkpoint 1.5**
> - In decode, what is the shape of `Q`? What is `seqlen_kv` and what
>   determines its value?
> - Why is decode harder for Tensor Cores than prefill?
> - What's the wall-clock split between prefill and decode for a chat
>   request that takes 512 input tokens and generates 256 tokens? (You
>   need a rough mental model — don't compute exactly.)

---

## 1.6 The KV cache — what we are caching and why

Naively, every decode step would re-read every prior token's
representation from the start, project it through the layer's QKV
weights, and use those K and V values. That re-projection is
**redundant** because the layer's K and V for prior tokens are *fixed*
once computed — they don't depend on the current decoding step.

So we **cache K and V** across decode steps. Each layer maintains a
per-batch buffer of all K and V values for all previously seen
positions, and just *appends* the new token's K, V to it each step.

That cache is the **KV cache**, and it has the shapes we've been seeing:

```
Per layer, per batch slot:
K : [n_kv_heads, max_seqlen, head_dim]   fp16
V : [n_kv_heads, max_seqlen, head_dim]   fp16
```

Why not cache Q? Q changes every step (it's the projection of the *new*
token's input), and we only use the current step's Q for the current
step's attention. So Q doesn't accumulate.

### The 64 GB problem

The cache size grows linearly with sequence length AND with batch size.
For Llama 3.1 8B:

```
bytes_per_token_per_layer = 2 (K, V) · n_kv_heads · head_dim · sizeof(fp16)
                          = 2 · 8 · 128 · 2
                          = 4096 bytes per token per layer
```

Across all `n_layers = 32` layers, that's `4096 · 32 = 131,072 bytes
per token = 128 KiB per token` per batch slot. Wait — the design doc
quotes 256 KB/token; let me redo the multiplication carefully:

```
2 (K and V) · 32 layers · 8 kv_heads · 128 head_dim · 2 bytes (fp16)
= 2 · 32 · 8 · 128 · 2
= 131,072 bytes = 128 KiB per token
```

(The docs/02 number of 256 KB/token uses `n_kv_heads=8` but counts each
of K and V as `n_kv_heads · head_dim · 2 = 2048 bytes`, then `2 · 32
layers · 2048 = 131072 bytes` — same answer. The "256 KB" in docs/02 is
slightly off; trust the calculation here.)

Now scale it up. For a chat-serving workload of `batch=32` users,
`seqlen=8192` tokens each:

```
total_kv_bytes = batch · seqlen · 128 KiB
               = 32 · 8192 · 131072
               = ~34 GiB
```

A 24 GB RTX 4090 doesn't fit that, never mind 16 users. **KV cache, not
weights, is what caps batch size and context length on consumer-class
hardware.**

This is the constraint Phase 2 attacks. INT8 KV cuts this in half.
INT4 KIVI cuts it to a quarter.

### KV cache and bandwidth

There's a second cost to the KV cache beyond memory: every decode step
reads **the entire** K and V from HBM into compute. For one (batch,
head) at `seqlen_kv = 4096`, fp16:

```
K bytes per (batch, head) = seqlen_kv · head_dim · 2 = 4096 · 128 · 2 = 1 MiB
V bytes per (batch, head) = same = 1 MiB
total per (batch, head) = 2 MiB
total for batch=8 × n_heads=32 (Q heads): 256 · 2 MiB = 512 MiB read per decode step
```

GQA reduces this — with `n_kv_heads=8`, each kv head's K, V is shared
by 4 query heads, so unique reads are `8 · 8 · 1 MiB · 2 = 128 MiB`,
not 512 MiB (modulo L2 cache behaviour). Still: every decode step is
moving a sizable chunk of data from HBM. **Bandwidth is the second
constraint** the KV cache imposes.

INT8 KV halves the bytes-per-element. INT4 quarters. That's the
*memory bandwidth* benefit of compression, separate from the *memory
capacity* benefit.

> **Checkpoint 1.6**
> - Why do we cache K and V but not Q?
> - At `batch=16, seqlen=4096`, how big is the fp16 KV cache for Llama
>   3 8B? Does it fit in 24 GB alongside the 16 GB of weights?
> - If we go from fp16 KV to INT8 KV, the memory savings are obvious.
>   What's the *bandwidth* savings during decode?

---

## 1.7 What does a fast decode kernel look like, then?

Tying it together. A *single decode step* for one (batch, query-head)
pair looks like this:

```
Inputs:
  q     : [head_dim]                    fp16    (one Q vector)
  K     : [seqlen_kv, head_dim]         fp16    (cached K for this kv head)
  V     : [seqlen_kv, head_dim]         fp16    (cached V for this kv head)
  scale : scalar  (1 / √head_dim)

Compute:
  s[j]    = scale · dot(q, K[j, :])         for j in 0..seqlen_kv-1
  m       = max(s)
  p[j]    = exp(s[j] − m) / Σ_j exp(s[j] − m)
  out[d]  = Σ_j p[j] · V[j, d]              for d in 0..head_dim-1

Output:
  out   : [head_dim]                    fp16
```

That's the entire decode attention for one query head, one batch slot,
at one decode step. To get the full per-token output of the model:

- For each of `n_layers = 32` layers,
  - For each of `n_heads = 32` query heads,
    - Run the above. Use the index map `kv(h) = h/4` to pick the right
      KV head.
- Concatenate the 32 head outputs into one `4096`-dim vector and apply
  `W_o`.
- Add the MLP block.
- Add the LayerNorm.

The attention kernel we write does **the inner loop** — one (batch,
head) pair's decode attention. It runs once per (batch, head) per layer
per decode step. Latency of this kernel multiplies by `n_layers ·
n_heads` × decode steps to give wall-clock cost.

### What success looks like

For a **kernel-level** win:
- **Latency**: as fast as PyTorch SDPA. SDPA dispatches to
  FlashAttention/cuDNN — the state of the art. On our reference
  workload, SDPA hits 1.36 ms; our v3 hits 0.713 ms — 1.91× faster.
- **Bandwidth**: meaningful fraction of HBM peak. Our v3 hits 189 GB/s
  of 1008 GB/s peak (19%). Not at peak, but well into useful range.

For a **memory** win:
- **KV bytes per token** as small as quality allows. fp16 is 4096 B; INT8
  is ~2080 B (incl. scales); INT4 KIVI is ~1080 B. We measured perplexity
  on WikiText-2 to confirm quality stays within budget.

For a **system** win:
- Both kernels integrated into a real LM serving stack, so the
  microbench gains translate to user-facing tokens/sec. That's Phase 4.

> **Checkpoint 1.7**
> - Sketch the decode attention computation in 5 lines, using `q, K, V,
>   scale` and a single output `out`. (You should be able to write this
>   from memory by now.)
> - In our project, what wraps the kernel into per-layer, per-head form
>   so a full decode step happens? (Hint: nothing yet for Phase 1/2 —
>   we benchmark the kernel in isolation. Phase 4 integrates it.)
> - How many times per decoded token does the attention kernel run for
>   Llama 3 8B with batch=8?

---

## 1.8 Summary of Part 1

You should now be comfortable with:

- **Attention** as a softly-weighted average of V's, indexed by Q-K
  similarity. The formula `O = softmax(Q · K^T / √d) · V`.
- **Multi-head** attention as `n_heads` independent attention computations
  on `head_dim = d / n_heads`-sized chunks.
- **GQA** as having `n_kv_heads < n_heads`, with the index map
  `kv(h) = h / (n_heads / n_kv_heads)`.
- **Prefill vs decode**: prefill is many queries in parallel,
  compute-bound; decode is one query against a growing KV cache,
  memory-bound, the "M = 1" problem.
- **KV cache**: cache K and V so we don't recompute them every step;
  size grows with batch × context; this is what caps serving throughput
  on consumer GPUs.
- **What the kernel does**: per (batch, head), score against all kv
  positions, softmax, weighted sum of V's. Returns one `[head_dim]`
  vector.

Everything from here on is about making this computation *fast*. Part
2 covers the GPU mental model: what hardware we have to work with, and
how performance is actually limited.

---

## 1.9 Glossary (for quick reference later in the book)

| Term | Meaning |
|------|---------|
| `d` (also "d_model") | Hidden size of the model (Llama 3 8B: 4096). |
| `n_heads` | Number of attention heads per layer (Llama 3 8B: 32). |
| `n_kv_heads` | Number of *KV* heads under GQA (Llama 3 8B: 8). |
| `head_dim` | `d / n_heads` (Llama 3 8B: 128). |
| `seq_len`, `seqlen_kv` | Number of tokens in context. |
| `Q`, `K`, `V` | Query, Key, Value tensors. |
| `Q · K^T` | The "score" matrix. |
| `softmax_scale` | `1 / √head_dim`. Stabilises softmax. |
| Prefill | Initial pass over the prompt (many query positions). |
| Decode | One generated token at a time (one query position). |
| KV cache | Stored K, V from prior decode steps. |
| GQA | Grouped Query Attention — fewer KV heads than Q heads. |
| HBM | High Bandwidth Memory — the GPU's main DRAM. |

---

**Ready for Part 2?** Part 2 (the GPU mental model) is where we build
the *performance* intuition: how threads/warps/blocks map to hardware,
the memory hierarchy, what __syncthreads costs, and the diagnostic
framework (bandwidth-bound vs compute-bound vs dependency-chain-bound)
that runs through every kernel we wrote.

Before continuing, try to answer at least the §1.6 and §1.7 checkpoints
without looking. Those two are load-bearing for the rest of the book.

---

# Part 2 — The GPU mental model

Part 1 told you *what* the kernel computes. Part 2 tells you *how the
hardware will run it* and, crucially, *what will limit how fast*. This
is the part of the book that you'll come back to most — it's the
toolkit for reading any kernel and knowing where the time is going.

## 2.1 Why a mental model?

When we write a CUDA kernel, what we're really doing is *describing a
parallel computation to the hardware*. The hardware then has a lot of
discretion in *how* it runs it — which threads execute when, which
loads come back when, which warps run on which SM. To make a kernel
fast, we need a working theory of what the hardware will do with our
code.

The frustrating-but-honest truth: there are many ways a kernel can be
"slow," and you can't tell which one applies just by looking at the
code. You have to *measure*, then *diagnose*. The mental model is what
lets you connect a measured number (e.g. "0.713 ms / 189 GB/s") to a
specific bottleneck (e.g. "we're not bandwidth-limited; the per-iter
dependency chain is").

By the end of Part 2 you should be able to:

1. Read a kernel and predict roughly how many warps will run per SM.
2. Look at a per-iter loop body and identify what the longest serial
   dependency chain through it is.
3. Hear "this kernel hits 189 GB/s of 1008 peak" and know whether
   that's good or bad given the workload.
4. Recognise the three diagnostic categories — bandwidth-bound,
   compute-bound, dependency-chain-bound — and know which
   optimizations target which.

These are the skills behind every commit in Phase 1, Phase 2, and Phase
3, even when we didn't say so explicitly.

> **Checkpoint 2.1**
> Before reading further: when a kernel achieves "20% of peak HBM
> bandwidth," does that mean it's fast, or slow? What would you need
> to know to decide?

---

## 2.2 The hardware: SMs, warps, threads

The RTX 4090 has **128 Streaming Multiprocessors (SMs)**. Each SM is a
mostly-independent compute unit with its own register file, shared
memory, L1 cache, and four "warp schedulers." All 128 SMs share the L2
cache and HBM.

```
GPU
 ├── 128 SMs
 │    ├── 4 warp schedulers           (issue 1 warp's instruction per cycle each)
 │    ├── 65536 32-bit registers      (split among resident threads)
 │    ├── 100 KB shared memory + L1   (split between the two, configurable)
 │    └── compute units               (CUDA cores for fp32/int, Tensor Cores for MMA)
 │
 ├── L2 cache                          (72 MB total, shared across all SMs)
 └── HBM (main DRAM)                   (24 GB, ~1008 GB/s peak bandwidth)
```

When you launch a kernel like:

```cuda
dim3 grid(batch, n_heads);   // 8 × 32 = 256 blocks
dim3 block(32);              // 32 threads per block
my_kernel<<<grid, block>>>(...);
```

…the GPU schedules the 256 blocks onto the 128 SMs. Each block stays on
the SM it's assigned to until it finishes (no migration). Multiple
blocks may run on the same SM at once if they fit (more on this in
§2.5).

Inside a block, threads are organised into **warps of 32 threads**.
A block of 32 threads = 1 warp. A block of 128 threads = 4 warps. A
block of 1024 threads = 32 warps. Warps are the fundamental unit of
execution: the SM doesn't issue instructions per-thread, it issues
*per-warp* (one instruction at a time across all 32 threads).

### Why 32?

Hardware history. The original GPUs had SIMD-32 execution units, and
the abstraction stuck. Modern NVIDIA GPUs all have warp size 32.
AMD GPUs have warp size 64 (they call them "wavefronts"). The number
matters for our purposes because:

- Warp shuffles operate on 32 lanes (`__shfl_xor_sync`, etc.).
- A "warp-wide" memory load that's coalesced reads up to 32 elements
  per instruction.
- Block sizes are usually multiples of 32 so no warp has idle lanes.

The single most important fact about a warp: **all 32 lanes execute
the same instruction in the same cycle**. That's *SIMT* — Single
Instruction, Multiple Threads. We'll get back to what happens when
the lanes diverge in §2.4.

### Mapping the kernels we wrote

Here's how the actual Phase 1 attention kernel `v3` maps to this
hierarchy:

```
Launch: grid(batch=8, n_heads=32) = 256 blocks
        block(32) = 32 threads = 1 warp per block

Each block:
  - Runs on one SM. With 256 blocks and 128 SMs, ~2 blocks per SM.
  - Holds 1 warp (32 threads) — single-warp block.
  - Owns one (batch, head) pair: reads q [head_dim], K and V
    [seqlen_kv, head_dim], writes out [head_dim].

Each thread (= one lane within the warp):
  - Owns 4 d-lanes (since head_dim=128, 128/32 = 4).
  - Iterates the seqlen_kv loop, accumulating one output.
```

For Phase 3 v3 (Phase 3c decode GEMM), the block has 128 threads
(4 warps) and they share the K-reduction work — different kernel
structure, same building blocks.

> **Checkpoint 2.2**
> - For a kernel launched with `grid(8, 32), block(32)`, how many warps
>   total are dispatched to the GPU?
> - On the RTX 4090's 128 SMs, what's the *minimum* number of SMs that
>   would have to be active to host that workload? (Hint: at least one
>   warp per used SM, but multiple blocks can share.)

---

## 2.3 The memory hierarchy

If you internalise one thing from Part 2, make it this: **memory is a
hierarchy with latencies that span 3–4 orders of magnitude**. Where
your data lives determines whether your kernel runs in microseconds or
milliseconds.

The hierarchy, fastest to slowest, on RTX 4090:

| Storage      | Scope               | Size         | Read latency (rough) | Bandwidth |
|--------------|---------------------|--------------|---------------------:|----------:|
| Registers    | Per-thread          | 255 × 32-bit | 0 cycles (free)      | n/a       |
| Shared mem   | Per-block           | 100 KB / SM (configurable) | ~20 cycles | ~10 TB/s/SM |
| L1 cache     | Per-SM              | shares 100 KB with shmem | ~28 cycles | ~10 TB/s/SM |
| L2 cache     | Per-GPU             | 72 MB          | ~190 cycles        | ~5 TB/s total |
| HBM (DRAM)   | Per-GPU             | 24 GB          | ~500 cycles        | 1008 GB/s peak |

The actual cycles vary by access pattern, contention, cache state, and
fp16 vs int32 vs ... — these numbers are ballpark. But the *ratios*
matter: HBM is ~25× slower latency than shmem, and ~100× slower than
registers.

A working analogy: **think of HBM as a slow truck and shmem as a fast
conveyor belt right next to your workbench**. You don't move things off
the workbench unless you have to. You load big batches from the truck
once and then work from the conveyor belt.

### What "bandwidth-bound" actually means

When we say a kernel is "bandwidth-bound on HBM," we mean: the runtime
is set by *how fast we can pull data through the HBM-to-SM pipe*, and
no amount of compute optimization will help unless we cut HBM bytes.
The 4090's HBM peak is 1008 GB/s. If our kernel needs to move 128 MB
through HBM and that takes 0.127 ms, we're at HBM peak. If it takes
0.7 ms, we're at 18% of HBM peak — meaning *something else is the
ceiling* and we have headroom.

The cheat-sheet question: **"if we doubled the HBM bandwidth, would
the kernel run twice as fast?"** If yes, bandwidth-bound. If no,
something else.

### Where each thing lives in our kernels

For the Phase 1 v3 decode attention kernel:

```
q   (one [head_dim] = 256 bytes per (batch, head))   →  loaded into per-thread REGISTERS once at start
K   (whole [seqlen_kv, head_dim] = 1 MB per block)   →  streamed from HBM via L1/L2
V   (whole [seqlen_kv, head_dim] = 1 MB per block)   →  streamed from HBM via L1/L2
o_acc, m, l (per-thread running state)               →  REGISTERS for the whole loop
out (one [head_dim])                                 →  written to HBM at end (one vec store)
```

Q lives in registers because it's reused over every j iteration. KV is
read once and not reused, so streaming from HBM is fine — we don't
benefit from caching it. The running softmax state stays in registers
because per-iter updates are critical-path.

For the Phase 2b INT8 KV attention, the structure is the same but K
and V live as int8 in HBM (half the bytes vs fp16). For Phase 3c W4A16,
the activations are *cached in shared memory* because the same K
values are reused N times across the output columns the block
computes — that reuse is what makes shmem caching worth it.

> **Checkpoint 2.3**
> - In the v3 decode attention kernel, why is Q held in registers
>   rather than streamed from HBM?
> - In the Phase 3c W4A16 GEMM kernel, why is `act` worth caching in
>   shared memory but `weight` not?
> - The 4090's HBM peak is 1008 GB/s. If a decode kernel reads 128 MB
>   of KV per call, what's the *theoretical fastest* it could run?

---

## 2.4 SIMT execution: same instruction, 32 lanes

The single-instruction-multiple-thread model is the most important
performance abstraction in CUDA, and it has *consequences*.

### What "lockstep" means

When a warp executes an instruction like `c = a + b`, all 32 lanes
execute that instruction in the same cycle. Each lane has its own
copy of `a`, `b`, `c` in its registers — the instruction operates on
32 register copies simultaneously.

So if the warp does:

```cuda
float a = some_value;        // lane 0 has its own, lane 1 has its own, ...
float b = other_value;
float c = a + b;             // all 32 lanes add, in one cycle
```

The work is "free" in the sense that lane 1 doing this work doesn't
make lane 0's work slower. The warp's compute throughput is *all 32
lanes at once*.

This is why we keep saying "**redundant scalar work is essentially
free under SIMT**." If lane 0 needs to compute `m_new = max(m, s_j)`,
and we have 32 lanes already executing the same warp instruction, then
having all 32 lanes redundantly compute the same `m_new` is free.
Lanes 1–31 weren't going to do anything more useful on that cycle
anyway. (This was the Phase 1 v1 "wrong hypothesis" lesson: trying to
*save* this "redundant" work by concentrating it in one lane introduced
serialization where there was none.)

### Divergence: when lanes don't agree

What if the lanes need to do *different* things? Consider:

```cuda
if (threadIdx.x < 16) {
    a = x + y;
} else {
    a = x - y;
}
```

In the warp, the first 16 lanes want `x + y`; the other 16 want `x - y`.
The hardware can't execute two different instructions in one cycle.
Instead, it executes the first branch with only lanes 0–15 *active*
(lanes 16–31 are masked off — their results aren't written), then the
second branch with lanes 16–31 active. The cost: **2× the cycles**
for that block of code.

Divergence is most painful when it's complex or unpredictable.
Predictable divergence (e.g. one branch is rare) costs less because the
"both branches" cost is amortized.

For our kernels, we mostly *avoid* divergence by design:
- Bounds checks at the top of the kernel (`if (n >= N) return;`)
  diverge but only at the edges, so most blocks aren't affected.
- The per-iter loop body has no `if`s that depend on per-thread state —
  every lane does the same dot-product, the same softmax update, the
  same FMA, just on its own data.

### Warp shuffles: cross-lane communication

Within a warp, lanes can exchange register values without going through
shared memory using **warp shuffles**:

```cuda
float partial = q_val * k_val;
partial = __shfl_xor_sync(0xFFFFFFFF, partial, 16);   // swap with lane^16
partial = __shfl_xor_sync(0xFFFFFFFF, partial, 8);    // swap with lane^8
... // continue with offsets 4, 2, 1
// Now all 32 lanes have the same value: the sum across the whole warp.
```

A shuffle is a "butterfly" pattern: with 5 shuffle ops (offsets 16, 8,
4, 2, 1), every lane ends up with the sum (or max, etc.) across the
whole warp. This is *much* cheaper than going through shared memory
because there's no shmem store + sync + load — it's all register-to-
register over the warp's internal datapath. **5 cycles vs ~30+ for
shmem reduction.**

Our kernels use this in `warp_reduce_sum` and `warp_reduce_max`. The
v3 attention kernel's entire dot-product reduction is one
`warp_reduce_sum` call — no shmem involved.

> **Checkpoint 2.4**
> - When 16 lanes take a branch and 16 don't, how many cycles does
>   the branch cost compared to all 32 taking the same path?
> - `warp_reduce_sum` does 5 shuffle operations. Why exactly 5? (Hint:
>   what powers of 2 are involved?)
> - In Phase 1 v3, we said "redundant scalar work across the warp is
>   free." Give a concrete example from a kernel where this principle
>   matters.

---

## 2.5 Occupancy and latency hiding

Threads and warps don't run continuously — they spend a lot of time
*stalled*, waiting for memory loads to complete, waiting for the
result of an `__expf`, etc. The GPU's main strategy for hiding these
stalls is to keep many warps resident on an SM and switch between
them when one stalls.

### What "resident" means

An SM has a finite budget — registers, shared memory, warp slots.
When a block is launched onto an SM, it *uses* some of that budget:

```
Resources used by one block of K threads, R registers/thread, S bytes shmem:
  registers    : K · R                  (out of 65536)
  shared mem   : S                      (out of 100 KB)
  warp slots   : ceil(K / 32)           (out of 48 warps/SM on Ada)
  block slots  : 1                      (out of 16-24 blocks/SM on Ada)
```

The SM holds as many blocks as fit under all four constraints. The
**occupancy** is the resulting `active warps / max warps`:

```
occupancy = (blocks_per_SM · warps_per_block) / 48
```

For v3 (single-warp block, 33 registers/thread): the constraint is the
block-slot limit of 16 (or whatever the Ada hard cap is on small
blocks). 16 blocks × 1 warp = 16 warps. Occupancy = 16/48 ≈ 33%.

For Phase 1 v0 (128-thread block, 29 registers/thread): up to 12 blocks
× 4 warps = 48 warps = 100% occupancy.

### How occupancy helps

When a warp issues a memory load that misses in cache, it takes
hundreds of cycles to come back. During those cycles, the warp scheduler
looks at the *other* resident warps and picks one whose next
instruction is ready to issue. If there's no ready warp, the SM stalls.

With many resident warps, there's almost always *some* warp ready, so
the SM keeps issuing instructions. **High occupancy = latency hiding
via parallel waiting**.

But there's a saturation: once you have enough warps that the SM is
never stalled waiting, more warps don't help. Past that point, occupancy
is just overhead (register pressure, more shmem competition, etc.).

### When occupancy matters and when it doesn't

It matters when the kernel has lots of memory-load latency to hide. If
every loop iteration has a cache-missing load with a 500-cycle latency,
you want enough warps that the SM can do ~16 instructions per cycle
across them while one waits.

It *doesn't matter* when the per-iter dependency chain is the bottleneck.
If each warp's next instruction depends on the previous one (true
serial), no amount of additional warps helps — the SM still issues
4 instructions per cycle, but one warp's instructions can only proceed
in sequence.

This was the Phase 1 v3 lesson: we traded 4 warps/block (full occupancy)
for 1 warp/block (33% occupancy) and *gained* 1.50× speedup. The lost
latency hiding didn't matter because the per-iter chain was the
ceiling. **Occupancy is a means, not an end.**

### How to read the bench output

When we say v3 is at "33% occupancy" we mean: of the 48 warp slots
each SM could hold, only 16 are active. The remaining slots are wasted,
but that's not the same as a slow kernel — it's only slow if those
extra warps would have helped hide latency, which they don't here.

> **Checkpoint 2.5**
> - A kernel uses 60 registers per thread. With 65536 registers per SM,
>   how many threads can it have resident? (Ignore other constraints.)
> - Phase 1 v3 has 33% occupancy and is faster than v2 at 100%
>   occupancy. Explain this in one sentence.
> - When would *adding* warps make the kernel slower?

---

## 2.6 Synchronization: what __syncthreads costs and means

When threads in different warps need to coordinate — write to shmem
in one warp, read in another — you need a barrier. The two main
primitives:

### `__syncthreads()`

Block-wide barrier. **Every thread in the block must reach the
`__syncthreads` before any thread proceeds past it.** Internally:

- The SM tracks how many warps have hit the barrier.
- Warps that arrive early are parked (the scheduler picks other
  warps).
- When all warps in the block have hit, all are released.

Cost: typically tens of cycles (it depends on how synchronised the
warps were going in). For our kernels, the bigger cost is usually the
*consequence* — the compiler can't reorder loads or stores across a
`__syncthreads()`. That's a load barrier, not just a sync barrier.

This is the Phase 1 v1 lesson: V loads inside the per-`j` `__syncthreads()`
couldn't be hoisted by nvcc above the sync, so V latency couldn't hide
behind the K reduction. The fix was to manually move the V load to the
top of the iteration — same code, just outside the barrier.

### `__syncwarp()`

Warp-wide barrier (cheaper). For single-warp blocks, mostly redundant
because the lanes are already in lockstep. But there are subtle cases
(e.g., after writing to shmem from one lane, before another lane reads
it) where you need an explicit warp sync to defeat compiler reordering.

### Warp shuffles as cross-lane sync

The `__shfl_xor_sync` calls we discussed in §2.4 don't need a separate
sync — the "sync" is implicit in the shuffle's mask (the first arg,
typically `0xFFFFFFFF` to require all 32 lanes participate). This is
the cheapest way to communicate across a warp.

### A subtlety: __syncthreads() and divergence

`__syncthreads()` deadlocks if some threads in the block can't reach it
(because they returned early, for example). For this reason, you'll
see patterns like:

```cuda
// BAD — deadlocks if some threads return early
if (n >= N) return;
... do work ...
__syncthreads();
```

```cuda
// OK — all threads reach the sync
const bool valid = (n < N);
... do work guarded by valid ...
__syncthreads();
if (valid) { ... }
```

We hit this in the Phase 3c kernel where some threads might be out of
the N range. The fix was to guard with `valid` and have all threads
participate in the sync.

> **Checkpoint 2.6**
> - You write a kernel where every thread writes to its own shmem
>   slot, then every thread reads that same slot back. Do you need
>   `__syncthreads()`?
> - In Phase 1 v1, why does putting the V load *before* the
>   `__syncthreads()` matter for performance?
> - What goes wrong if `__syncthreads()` appears inside an `if` branch
>   that only some threads take?

---

## 2.7 The dependency chain

This is the concept that *most* of our optimizations turn on, and the
one that's least taught in a 101 course.

### What "dependency chain" means

In a per-iteration loop body, there's often a sequence of operations
where each depends on the previous:

```
load → unpack → multiply by scale → FMA → softmax update → next iter
```

Each of those operations has some *latency* — the number of cycles
from issuing the instruction to having its result available. The
*total* latency through the chain is the sum of the latencies.

If the chain is short, the next iter starts soon and throughput is
high. If the chain is long, the next iter waits.

### The "instruction issue rate" view

The SM can issue ~4 warp instructions per cycle (one per warp
scheduler). If the warps have lots of *independent* work, the SM is
saturated — every cycle, 4 ops are issued. That's "compute-bound."

But if the warps' work is *serially dependent* (instruction N depends
on instruction N-1's result), then each warp can only have one
instruction in flight at a time, and the warp issues at the rate of
the chain (1 / chain_length per warp-cycle).

The SM can compensate by switching between many warps — each warp has
its own chain, and the SM rotates through them. With enough resident
warps, the SM stays saturated even if each individual warp's chain is
long. *This is what occupancy buys you.*

When does that not work?

- If each warp has only a tiny number of independent ops, you need
  lots of warps to fill the SM. The kernel becomes occupancy-limited.
- If memory loads are part of the chain, those add hundreds of cycles
  of latency — even more warps needed.

### "Dependency-chain-bound" as a category

This is the third category beyond bandwidth-bound and compute-bound:
the kernel is bottlenecked by **how fast the per-iter chain can
complete**, regardless of how much compute or bandwidth is available.

How to recognise it:
- HBM is not at peak (so not bandwidth-bound).
- Compute units are not at peak (so not compute-bound).
- The kernel doesn't speed up with more SMs (Phase 1 v4) or more
  parallel loads (Phase 1 v5).
- The kernel *does* speed up when you make the chain shorter (Phase 1
  v3's single-sync reduce, Phase 2c's per-group K scale).

Phase 1 v3 → v2: the gap was a `__syncthreads()` in the middle of the
chain. Removing it cut the chain. **The 570 µs we measured wasn't
"sync overhead" — it was "what the chain was waiting for."**

Phase 2c: by moving K scales out of the per-iter loop (loading them
once per group instead of every iter), we cut a load+multiply from the
chain. The structural shorter-chain made the kernel faster, not the
fewer bytes loaded.

### The diagnostic question

For any kernel, ask: **what's the per-iter critical path?** Then ask:
**what would shorten it?**

For decode attention: the per-iter chain is `load K` → `dot product +
warp_reduce_sum` → `softmax recurrence` → `FMA with V` → `next iter`.
Each step is a few cycles or more (warp_reduce_sum is ~5 cycles for
the shuffle tree). Total per-iter chain: ~30-50 cycles, depending on
loads and dependencies.

For W4A16 GEMM: the per-iter chain is much shorter (no softmax, no
reduction within a thread — just FMA). That's why W4A16 wins big on
memory-bound shapes: it has no chain ceiling and the byte savings
translate directly.

> **Checkpoint 2.7**
> - What's the per-iter dependency chain in the v3 attention kernel?
> - Why did the "single-sync reduce" in v2 produce *more* speedup
>   than just removing a `__syncthreads()` would predict?
> - Phase 2b (INT8 KV) halved KV bytes but didn't speed up the
>   kernel. Where was the chain hiding the win?

---

## 2.8 The diagnostic framework

You now have the vocabulary to talk about where a kernel's time goes.
The framework:

### Three bottleneck categories

1. **Bandwidth-bound**: the kernel is reading data through HBM at
   close to peak (or close to L2 peak, or whatever the relevant
   memory level is). Cutting bytes helps. Adding more SMs doesn't.
   *Diagnostic*: achieved bandwidth ≈ peak.

2. **Compute-bound**: the kernel is doing arithmetic close to the
   GPU's FLOPS peak. Adding more bandwidth doesn't help. Tensor Cores
   (if applicable), wider vectorization, or fewer ops do.
   *Diagnostic*: achieved TFLOPS ≈ peak compute.

3. **Dependency-chain-bound**: neither of the above. The per-iter
   serial chain limits throughput. Shorter chains (fewer ops per iter,
   moving work out of the loop) help. More SMs / more bandwidth
   don't.
   *Diagnostic*: well below both HBM and compute peaks, kernel doesn't
   speed up with bigger grid or wider loads.

For decode attention on the 4090:
- v3 at 189 GB/s of 1008 GB/s = 19% of HBM peak. NOT bandwidth-bound.
- Compute-wise, attention's per-iter ops are tiny. NOT compute-bound.
- Therefore dependency-chain-bound. The optimization lever was *the
  chain*: v3 cut the cross-warp shmem broadcast (v2 → v3), v3 used
  vectorized loads + single warp to keep the chain compact, v4/v5
  failed because they didn't shorten the chain.

For W4A16 GEMM on the 4090 at M=1:
- 1577 GB/s of 1008 GB/s = 156% of HBM peak (L2-served). Effectively
  bandwidth-bound on L2.
- The kernel's win comes from cutting weight bytes 4× (fp16 → INT4).
- Compute is trivial.

### How to apply the framework

When you measure a new kernel and it's slower than you hoped:

1. **Calculate achieved bandwidth** from bytes moved / time. Compare
   to peak. If close to peak: bandwidth-bound. Cutting bytes is your
   tool.
2. **Calculate achieved compute** (FLOPS / time). Compare to peak.
   If close: compute-bound.
3. **If neither**: dependency-chain-bound. Look at the per-iter loop
   body and find what depends on what. The chain is your enemy.

When picking an optimization:
- Don't add bandwidth-targeting work to a kernel that isn't bandwidth-
  bound. (Phase 1 v4 split-K, Phase 1 v5 cp.async — both lost.)
- Don't add compute-targeting work to a kernel that isn't compute-
  bound. (Tensor Cores at decode M=1 — wrong tool.)
- *Do* attack the chain — but recognise what's in the chain (loads,
  shmem hops, reductions, `__expf`, etc.).

---

## 2.9 Putting it together: reading a kernel like a performance engineer

A working checklist for sizing up any GPU kernel:

**1. Block geometry**
- `grid` size, `block` size, total warps?
- Reasonable for the GPU? (e.g. 256 blocks of 4 warps each on 128 SMs
  is fine; 32 blocks of 1 warp each is severely under-occupied.)

**2. Per-thread work**
- What does each thread own? Registers used?
- Does each thread loop?

**3. Memory pattern**
- What's read from HBM, how many times?
- What's cached in shmem, L1, registers?
- Coalesced loads across the warp?

**4. Per-iter critical path**
- What's the longest serial dependency through one loop iter?
- How many cycles does it take?

**5. Synchronization**
- `__syncthreads()` calls — how many per iter? Why?
- Any places where loads sit *inside* a sync that could be hoisted out?

**6. Diagnostic guess**
- Bandwidth, compute, or chain bound?
- What evidence (from the code) supports that guess?
- What's the right optimization given the guess?

Try applying this checklist to your favorite of the v0–v5 attention
kernels before reading on. (Hint: they all have different answers for
items 1, 4, and 5.)

---

## 2.10 Summary of Part 2

By the end of Part 2 you should be comfortable with:

- The hierarchy: **GPU → SMs → blocks → warps → threads**, and how a
  kernel launch maps to this.
- **SIMT execution**: 32 lanes in lockstep, divergence costs, why
  redundant warp-wide work is free.
- **Memory hierarchy**: registers / shmem / L1 / L2 / HBM, with
  latencies spanning 3–4 orders of magnitude. Where to put what.
- **Occupancy and latency hiding**: more resident warps → more
  in-flight work → better at hiding memory latency. Capped by
  register/shmem/warp budgets.
- **Synchronization**: `__syncthreads()` is a barrier *and* a load
  barrier. Warp shuffles are cheaper than shmem reductions.
- **The dependency chain**: per-iter serial path that limits
  throughput when neither bandwidth nor compute is at peak.
- **The diagnostic framework**: bandwidth-bound vs compute-bound vs
  dependency-chain-bound — and how to tell which applies.

Part 3 will put this to work: we'll walk through the v0 → v5 attention
journey, applying these tools at each step. By the end you should be
able to predict *why* an optimization will or won't help, before
measuring.

---

## 2.11 Glossary additions

| Term | Meaning |
|------|---------|
| SM | Streaming Multiprocessor. The 4090 has 128. |
| Warp | 32 threads executing in SIMT lockstep. |
| Lane | One thread within a warp; `threadIdx.x % 32`. |
| Block | Set of warps that share an SM and shared memory. |
| Grid | Set of blocks. The kernel launch dispatches the grid. |
| Resident warp | A warp the SM is currently holding (vs queued or finished). |
| Occupancy | `active warps / max warps per SM`. |
| HBM | High Bandwidth Memory. The GPU's main DRAM. |
| Coalesced load | A warp-wide load where the 32 lanes' addresses are contiguous, served as one wide transaction. |
| Bandwidth-bound | Kernel runtime is set by data-movement throughput. |
| Compute-bound | Kernel runtime is set by arithmetic throughput. |
| Dependency-chain-bound | Kernel runtime is set by the per-iter serial dependency path. |
| SIMT | Single Instruction, Multiple Threads. NVIDIA's execution model. |
| Divergence | Lanes within a warp taking different branches; serializes the branches. |
| Warp shuffle | Register-to-register communication across lanes within a warp. |
| `__syncthreads()` | Block-wide barrier. Also prevents the compiler from reordering loads across it. |
| Latency hiding | Using other warps' work to fill in time while one warp waits on memory. |

---

**Ready for Part 3?** Part 3 walks the v0 → v5 attention kernel
evolution using Part 2's framework. Each kernel gets a paragraph or
two of "here's what the bottleneck was at this step, here's the lever
we pulled, here's why it worked (or didn't)."

Before continuing, work through the checkpoint questions in §2.5,
§2.7, and §2.8 without looking — those three are the load-bearing
ones for Part 3.
