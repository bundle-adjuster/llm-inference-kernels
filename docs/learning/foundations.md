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
