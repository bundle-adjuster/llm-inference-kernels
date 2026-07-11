# Phase 2 Journey — KV-Cache Compression

> This document complements [`02-kv-cache-compression.md`](02-kv-cache-compression.md)
> (the forward-looking design) and [`results/RESULTS.md`](results/RESULTS.md)
> (the numbers). It captures the **journey**: what each step's theory was,
> what we built, what we measured, and what we learned — especially when
> the result surprised us.
>
> Phase 2 was unusually rich in surprises. Three findings overturned a
> prediction we made one step earlier — they're called out below.

## Setting

- **GPU**: RTX 4090 (sm_89), 1008 GB/s peak HBM, 72 MB L2.
- **Workload**: Llama 3.1 8B head config — `n_heads=32`, `n_kv_heads=8`,
  `head_dim=128`. Microbench at `batch=8, seqlen_kv=4096`, fp16. Model-level
  perplexity on WikiText-2 test (131,008 tokens, 64 chunks × 2048 tokens).
- **What we measure**: KV-cache memory (incl. scale overhead); kernel
  latency (CUDA events, 25 warmup + 100 timed); kernel-level mean
  relative error vs the fp32-reference attention; model-level **Δppl**
  vs fp16-KV baseline on WikiText-2.
- **What we're racing**: the Phase 1 v3 kernel (0.713 ms / 189 GB/s on
  this workload, 1.91× faster than PyTorch SDPA). Phase 2's job is to
  keep that latency or beat it, and slash KV-cache memory.

> **⚠ Phase 7 / ✅ Phase 8:** The "1.91× faster than PyTorch SDPA" framing
> is **retired**. That baseline fed SDPA a 4×-expanded GQA KV cache; against
> GQA-native SDPA (`F.scaled_dot_product_attention(..., enable_gqa=True)`)
> the v3 kernel is actually **4.55× slower** (0.22× on its reference
> workload), because v3's single-warp block is **occupancy-bound**
> (~2 of 128 SMs), not bandwidth-bound. The v6 FlashDecoding split-K kernel
> (multi-warp blocks) is what finally beats fair SDPA — 155.6 µs vs SDPA's
> 157.3 µs (1.01×) at batch=8/kv=4096, 4.59× over v3, ~82% of peak HBM.
> See [`05-baseline-correction-journey.md`](05-baseline-correction-journey.md)
> (the correction) and [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md)
> (the fix). **The INT8/INT4 latency comparisons below are all vs OUR v3 and
> remain valid** — only the vs-SDPA framing changes.

Phase 2's primary metric is **memory**: a chat-serving KV cache at 8K
context × batch 32 is ~64 GB in fp16, so KV-cache is what caps both
concurrent users and context length. The decode kernel also reads the
KV cache in full every step, so KV bytes are a bandwidth cost too. The
docs/02 design predicted that 4-bit KV would make decode attention not
just smaller but **faster** because it's memory-bound. That prediction
ended up half-right, half-wrong, in an instructive way.

---

## Phase 2a — PyTorch reference + tests  ·  commit `3114309`

### Theory

Before any CUDA kernel, build a pure-Python correctness oracle: symmetric
integer quantize / dequantize, in two axis modes that cover the KIVI
recipe.

- **Per-token**: one fp16 scale per `(batch, kv_head, token)`, shared
  across `head_dim`. Natural for V — V doesn't have persistent per-channel
  outliers, so one scale per token suffices.
- **Per-channel groupwise**: one fp16 scale per
  `(batch, kv_head, group_of_tokens, head_dim_channel)`, where groups
  partition the seqlen axis (`group_size=32` per KIVI). Natural for K
  — K has persistent per-channel outliers, and groups give spatial
  locality so scales adapt as the cache streams.

### Design choices

- **Symmetric integer**: `qmax = 2^(bits-1) − 1` (127 for INT8, 7 for
  INT4). `scale = absmax / qmax`, clamped to ≥ 1e-8 to avoid the all-zero
  edge case. `q = clamp(round(x / scale), −qmax, qmax)`.
- **Reference stores 4-bit values in int8 containers** (no packing) for
  clarity; CUDA path packs 2 nibbles per byte.
- **KIVI / INT8 presets**: `quantize_kv_kivi_int4` (K per-channel
  groupwise, V per-token, bits=4) and `quantize_kv_int8_per_token` (both
  K and V per-token, bits=8) — the exact configurations the CUDA kernels
  mirror in 2b/2c.

### Result

22 round-trip tests pass: dequant(quant(x)) reconstructs x within the
bits-determined tolerance (1.0/qmax of the per-axis max). Configurations:
- `(bits ∈ {4, 8}) × (axis: per-token, per-channel-groupwise) ×
  (group_size ∈ {32, 128}) × (batch, seqlen)`.
- Includes `seqlen=100, group_size=32` to exercise the short-final-group
  code path.

Storage ratios verified: INT8 lands at 0.51× fp16, packed INT4 at
0.26–0.32× depending on scale density.

### What we learned

- The reference is the contract every CUDA kernel must match — having
  it in place first meant 2b/2c never had to debug "is the algorithm
  right or is the kernel wrong?" Two-way certainty.
- Test infrastructure subtlety: when `seqlen` isn't a multiple of
  `group_size`, the last group is short. Both the quantization and the
  per-(group, channel) tolerance assertion had to pad consistently before
  reshape, or the comparison shapes mismatch silently.

---

## Phase 2b — INT8 per-token CUDA path  ·  commit `6df94ef`

### Theory

INT8 symmetric per-token. Both K and V get one fp16 scale per token,
shared across `head_dim`. The kernel sees half the K/V bytes (1 B vs 2
B per element) plus a small scale overhead. The decode attention design
doc predicted this should be **faster**: decode is memory-bound, so half
the KV traffic should roughly halve the kernel time.

### Design choices

- **Quantize kernel**: grid `(batch, n_kv_heads, seqlen)`, block 32
  (one warp). Each thread vectorizes 4 fp16 lanes (`head_dim/32 = 4`):
  per-thread absmax, `warp_reduce_max`, scale = max/127 (clamped), 4-byte
  int32 store, lane 0 writes the fp16 scale.
- **INT8 attention kernel** (built on v3): same body, K/V loads now
  `LDG.E.32` (4 B per thread, vs v3's 8). Scale-folding optimization:
  ```
  partial = q · k_int               (4 multiplies, int values)
  partial = warp_reduce_sum(partial)
  s_j     = partial · k_scale · softmax_scale     (one mul folded in)
  ── softmax ──
  p_j_v   = p_j · v_scale           (V scale folded into FMA coefficient)
  o_v[d]  = o_v[d] · alpha + p_j_v · v_int[d]    (FMA on int values)
  ```
  Saves 8 multiplies per iter vs naive dequant-everything. Sound because
  the dot product is linear: `dot(q, k_int · k_scale) = k_scale · dot(q,
  k_int)` for a scalar k_scale.

### Result

| Metric | Value |
|---|---|
| KV memory | **65 MiB · 0.51× of fp16** (63 MiB saved) |
| Kernel latency | **0.713 ms** — tied with v3 (0.713 ms) |
| Effective KV bandwidth | 96 GB/s (vs v3's 189 GB/s) |
| Quantize one-shot cost | 0.124 ms per K (per-token amortised in serving) |
| Kernel-level max abs diff | 1.1e-3 vs fp16 reference |
| Kernel-level mean rel err | 2.5% (on random gaussians) |

### **Surprise: latency tied despite halving KV bytes**

The design doc said decode attention is memory-bound, so half the KV
traffic → ~half the time. The reality: **0.713 ms either way**. We were
moving twice as many bytes per second in v3 (189 GB/s) as in INT8
(96 GB/s), both well under HBM peak (1008 GB/s).

**Diagnosis**: the decode kernel is **dependency-chain-bound**, not
bandwidth-bound. The per-`j` critical path is `warp_reduce_sum →
softmax (fmaxf + 2 expf) → FMA`. That shape doesn't change between
fp16 and int8 — and *that's* what limits throughput on this workload.
This was directly consistent with Phase 1's v4 (split-K) and v5
(`cp.async`) results: both attacked bandwidth, neither helped.

> **⚠ Phase 7 / ✅ Phase 8:** "Dependency-chain-bound, not bandwidth-bound"
> was itself incomplete. The real ceiling on v3 was **occupancy** — its
> single-warp block fills only ~2 of 128 SMs. Phase 1's v4 (split-K) failed
> because it kept v3's single-warp block; the fix was *both* split-K *and*
> multi-warp blocks (4 warps/block), delivered by v6 in Phase 8. v6 reaches
> ~82% of peak HBM and runs 4.59× faster than v3, beating fair GQA-native
> SDPA. The INT8-tied-with-v3 result here is a real CUDA-vs-v3 measurement
> and still holds; only the "this is the hard ceiling" reading is retired.
> See [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

So we updated the framing: **Phase 2b's win is memory, not latency.**
0.51× KV memory means ~2× longer context or ~2× larger batch in the
same VRAM budget. Same answer time. That's still a clear production
win — but for different reasons than the design doc anticipated.

### What we learned

- **"Decode is memory-bound" is workload-dependent**, not a law. At
  our shape (`batch=8, seqlen_kv=4096`), the kernel was nowhere near
  HBM peak. The instruction throughput / dependency chain was the
  ceiling, the same ceiling Phase 1 hit with v4 and v5.
- **Scale-folding via linearity is the right trick for per-token K**.
  Saves real cycles by not materialising the dequantized K in registers.

---

## Phase 2c — INT4 KIVI (per-channel K, per-token V)  ·  commit `cccdced`

### Theory

KIVI (Liu et al. 2024): K has persistent per-channel outliers; V doesn't.
At 4-bit, per-token K is the wrong recipe — the outliers dominate the
per-token scale, squashing the rest of the channel into a few quantized
values. KIVI quantizes K **per-channel** (one scale per `head_dim` lane)
with **groups** along the seqlen axis (so scales adapt as the cache
streams). V stays per-token. Packed 4-bit storage: 2 nibbles per byte.

### Design choices

- **Storage layout**:
  - `K_q: [batch, n_kv_heads, seqlen, head_dim/2]` int8 — each byte =
    `(q_lo & 0xF) | ((q_hi & 0xF) << 4)`
  - `K_scale: [batch, n_kv_heads, n_groups, head_dim]` fp16
  - `V_q: [batch, n_kv_heads, seqlen, head_dim/2]` int8 packed
  - `V_scale: [batch, n_kv_heads, seqlen]` fp16
- **Quantize kernels**:
  - K per-channel groupwise: grid `(batch, n_kv_heads, n_groups)`,
    block 32. Two passes — pass 1 builds per-thread per-channel absmax
    across `group_size` tokens; pass 2 quantizes each `(t, channel)`
    using those scales, packs 4 nibbles → uint16 store.
  - V per-token int4: same shape as INT8 V kernel but qmax=7 and packed
    output (uint16 per thread).
- **INT4 attention kernel**: same v3 structure, but a key structural
  change in the inner loop. Per-channel K scales can't be folded out
  of the dot product like a per-token scalar could (the scales differ
  per d-lane). BUT K scales are **constant within a group**, so we
  pre-scale q **once per group**:
  ```
  for g in 0..n_groups:
      q_scaled[d] = q_v[d] · k_scale[g, d]    (4 muls, once per group)
      for t in g·group_size .. (g+1)·group_size:
          k_int = load_int4x4(K_q[t, ...])    (one uint16 → 4 nibbles)
          v_int = load_int4x4(V_q[t, ...])
          partial = q_scaled · k_int          (4 muls, inner loop)
          ... softmax + V FMA with per-token v_scale folded ...
  ```
  The inner loop is 4 multiplies on int values — no per-iter K scale
  load, no per-iter K scale fold. K scales are register-resident through
  the group.

### Result

| Metric | Value |
|---|---|
| KV memory | **34.5 MiB · 0.27× of fp16** (93.5 MiB saved) |
| Kernel latency | **0.554 ms — 1.29× faster than v3 (0.713 ms)** |
| Effective KV bandwidth | 65 GB/s |
| Quantize one-shot cost | K 0.07 ms, V 0.12 ms |
| Kernel-level max abs diff | 2.3e-2 vs fp16 reference |
| Kernel-level mean rel err | 42% (on random gaussians) |

### **Surprise: INT4 actually moves latency. The 2b diagnosis needed an update.**

Phase 2b concluded the decode kernel was dependency-chain-bound — fp16
vs int8 tied, bandwidth wasn't the ceiling, end of story. Phase 2c
*breaks the tie*: INT4 KIVI runs 1.29× faster than v3.

**Why** — what changed in the inner-loop dependency chain:

- **INT8** loaded one fp16 K scale **per j** alongside the K_q load.
  That scale load sat on the dependency chain (you need k_scale to
  multiply with the dot product result before softmax). The chain
  couldn't hide it.
- **INT4 KIVI** loads K scales **once per group** (every 32 j-iters),
  holds them in registers, and pre-folds them into q. The per-`j`
  inner loop has **no K scale load at all**. It's just `LDG.E.16` K_q
  + `LDG.E.16` V_q + a per-token v_scale load + the unchanged
  arithmetic chain.
- The K_q and V_q loads also shrink from `LDG.E.32` (INT8) to
  `LDG.E.16` (INT4), so the inner-loop's load count and total load
  bytes both drop.

The 2b diagnosis was right — the dependency chain *was* the limiter.
2c **shortened the chain itself** by moving K-scale loads out of the
inner loop. That's why INT4 wins where INT8 tied. It's a structural
win (loop nesting) more than a numeric one (bit count).

### What we learned

- **Per-channel quantization at 4 bits is friendlier to a fast kernel
  than per-token at 8 bits**, *for this hardware and shape*, because
  per-channel scales can be loaded once per group and amortised across
  many inner-loop iters. Per-token scales must load each iter.
- **The KIVI trick is two-for-one**: per-channel K is the *quality* win
  (captures the outliers) and the structural pre-scale-q trick is the
  *latency* win. KIVI's paper sells it for quality; the kernel design
  carries the speed.
- **Dependency-chain analysis matters more than bandwidth analysis at
  these decode shapes.** Both Phase 1's v4/v5 and Phase 2b/2c tell the
  same story: bandwidth wasn't the ceiling, instruction throughput
  through the per-iter dependency chain was. Optimizations that
  *shorten* that chain (single-sync reduce in 2/v3, per-group K scales
  in INT4 KIVI) win. Optimizations that just reduce bytes don't.

---

## Phase 2d — WikiText-2 perplexity validation  ·  commit `4dc66b8`

### Theory

The success criteria from docs/02 are model-level, not kernel-level:
INT8 Δppl < 0.2 (threshold), INT4 KIVI Δppl < 0.5 (target). The
kernel-level mean rel err numbers from 2b/2c (2.5% INT8, 42% INT4 on
random gaussians) don't translate directly — they're worst-case on
i.i.d. uniform noise, and real LLM activations have structure
(per-channel K outliers; V distribution narrower than worst case; the
softmax's max-subtraction makes output ranking robust to per-element V
noise).

### Design choices

- **Patch `F.scaled_dot_product_attention`** to round-trip K, V through
  the PyTorch quantization reference before delegating to the real
  sdpa. The model sees exactly the noisy K, V it would receive from a
  compressed cache at decode time.
- **Use the PyTorch reference, not the CUDA kernels.** Faithful because
  the CUDA kernels match the reference (we proved that in 2b/2c
  correctness tests) — and the perplexity eval doesn't need CUDA-fast
  quantization, the model forward dominates.
- **Patch point: F module attribute.** `LlamaSdpaAttention.forward`
  looks up `F.scaled_dot_product_attention` at call time, so rebinding
  the module attribute intercepts every layer's attention without
  touching HF's GQA/RoPE/cache plumbing.
- **Modes evaluated**: fp16 (baseline), INT8 per-token K+V, INT4
  per-token K+V (the KIVI comparator — naive INT4), INT4 KIVI
  (per-channel K, per-token V).
- **Workload**: WikiText-2 test, concatenated to one stream, chunked
  into 64 × 2048-token windows (131,008 tokens total).

### Result

| Mode | ppl | Δppl | % delta | Target | Verdict |
|---|---:|---:|---:|---:|---|
| fp16 (baseline)            | 7.055 | —       | —     | —     | — |
| INT8 per-token K, V        | 7.056 | +0.0008 | +0.01% | < 0.2 | **PASS** |
| INT4 per-token K, V (naive)| 7.517 | +0.462  | +6.54% | —     | KIVI comparator |
| **INT4 KIVI**              | **7.252** | **+0.196** | **+2.78%** | **< 0.5** | **PASS** |

Both the threshold (INT8 < 0.2) AND the target (INT4 KIVI < 0.5) cleared
with margin.

### **Surprise: the 42% kernel-level mean rel err didn't tank perplexity.**

INT4 KIVI showed 42% mean rel err on random gaussians at the kernel
level. The model-level Δppl is +0.196 — 2.78%. **The kernel-level
worst-case number was ~15× pessimistic** about the model-level outcome.

Why the gap:

- **i.i.d. random gaussian K, V** is uniformly hard for symmetric
  per-token quantization. There's no structure for the scales to
  exploit; every channel has roughly the same magnitude, so the few
  bits of resolution must be spread across everything.
- **Real LLM activations have structure**. K's persistent per-channel
  outliers (the original motivation for KIVI) get their own scales,
  leaving the rest of the channel to be quantized well. V's distribution
  is much narrower than worst-case, so per-token int4 V actually loses
  little information.
- **Softmax max-subtraction is a forgiving operator.** The attention
  output is `Σ p_j · v_j`, where p_j comes from softmax of the dot
  products. As long as the *ranking* of the top scores survives
  quantization noise, the high-`p_j` tokens dominate the output and
  per-element V noise averages out. The model's downstream layers can
  also absorb a fair amount of activation noise.

### **The teaching moment: per-channel K is 2.36× better at the same INT4.**

At the same bit depth, INT4 KIVI's Δppl (+0.196) is **2.36×** better
than naive INT4 per-token K (+0.462). The naive version would have
flunked the < 0.5 target. KIVI passes with margin. Direct experimental
confirmation of the docs/02 prediction: K's per-channel outliers
genuinely need per-channel scales, and the cost is just a small scales
table — no extra bits per value.

### What we learned

- **Kernel-level worst-case error is a smoke test, not a quality
  verdict.** Re-evaluate at the model level before declaring an
  optimization too lossy.
- **Don't waste your bit budget on outliers.** Per-token quantization
  forces every channel to share scales, which is fine if magnitudes
  are uniform across channels (V) and wasteful if they aren't (K).
  The KIVI cost — `head_dim · n_groups` extra fp16 scales per kv-head
  — buys 2.36× quality at INT4. A great trade.
- **The patch-`F.scaled_dot_product_attention` trick is a clean way to
  prototype any "what if the cache were noisy" study** without touching
  HF's attention plumbing. Re-usable for future quantization experiments.

---

## Summary

| Step                                          | Memory      | Latency   | Δppl    | vs Phase-1 v3 (latency) | Notes |
|-----------------------------------------------|------------:|----------:|--------:|------------------------:|-------|
| fp16 KV (Phase 1 v3, baseline)                |  128.0 MiB  | 0.713 ms  | 0       | 1.00×                   | Phase 2's reference for everything below |
| INT8 per-token + fused dequant (2b)           | **65.0 MiB**| 0.713 ms  | **+0.0008** | tied               | Essentially lossless drop-in |
| INT4 per-token K, V (naive, 2d comparator)    |  ~35 MiB    | n/a       | +0.462  | n/a                     | KIVI comparator — fails the per-channel test |
| **INT4 KIVI** (per-channel K + per-token V, 2c+2d) | **34.5 MiB** | **0.554 ms** | **+0.196** | **1.29×** | The headline. Smaller, faster, AND quality budget met. |

### Lessons we want to carry forward

1. **"Memory-bound" is a workload property, not a kernel property.**
   Decode attention's design doc said it's memory-bound; on our
   `batch=8, seqlen=4096` workload at 1008 GB/s peak HBM, the kernel
   was at ~9–19% of peak. The dependency chain was the ceiling. Always
   verify which ceiling you're actually hitting before optimising
   against it (the same lesson Phase 1's v4 and v5 taught).

   > **⚠ Phase 7 / ✅ Phase 8:** The "dependency chain was the ceiling"
   > verdict was wrong about the *root cause*. Phase 8 found v3 was
   > **occupancy-bound** — one warp per block fills ~2 of 128 SMs, so the
   > kernel sat at ~18% of peak HBM regardless of the per-iter chain. The
   > v6 split-K kernel on multi-warp blocks hits ~82% of peak and runs
   > 4.59× faster than v3, which also flips the vs-SDPA verdict (v3 lost
   > 4.55×; v6 wins 1.01×). The INT8/INT4-vs-v3 comparisons in this doc are
   > unaffected. See [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

2. **Scale loop nesting, not just bit depth, decides whether
   quantization speeds the kernel up.** INT8 per-token loaded a scale
   per `j` and tied with v3. INT4 KIVI loads a scale per *group* and
   beats v3 by 1.29×. The win came from moving the scale load *out of*
   the inner loop, not from cutting bytes.

3. **Per-channel quantization at 4 bits is friendlier to the kernel
   than per-token at 8 bits, when the per-channel scales fold cleanly
   into a per-group pre-scale of q.** This was a surprise — we
   expected smaller bits = friendlier kernel, but it's the *loop
   structure* that wins, not the byte count.

4. **Kernel-level i.i.d. error is a worst-case smoke test, not a model
   quality verdict.** INT4 KIVI's 42% mean rel err on random gaussians
   would have killed the kernel had we taken it at face value. Real
   activations + softmax max-subtraction shrank that to a 2.78%
   perplexity delta. Always re-evaluate at the model level before
   declaring a compression scheme too lossy.

5. **The KIVI insight earns its name in two ways.** Per-channel K is
   the *quality* win (Δppl 0.196 vs 0.462 — 2.36× better) and per-group
   K scale loads are the *latency* win (1.29× over v3). The same
   structural trick attacks both axes.

6. **The kernel never holds dequantized K/V in HBM**, only in registers
   during the dot product / FMA. That's the "fused dequant" promise of
   docs/02 — the kernel reads packed bytes, dequantizes in registers,
   does attention, and never writes the fp16 form anywhere. Without
   this, INT4 would just move HBM bytes around without saving
   bandwidth.

## What's still open

- **Decode tokens/sec at the model level.** Phase 2's kernel-level
  bench shows the steady-state attention call is 1.29× faster with
  INT4 KV than fp16. Wiring that into Llama 3.1 8B's actual decode
  loop is Phase 4 integration work (custom KVCache subclass that
  stores INT4 packed bytes + per-channel/per-token scales, calls
  `quantize_*` on append, calls `decode_attention_int4` on read).
- **Quality eval beyond perplexity.** WikiText-2 ppl is one number;
  a chat-style eval (MMLU / GSM8K / a small instruction-following set)
  on the Instruct model would give a more "does it still feel right"
  read. Especially relevant since the Δppl gap between fp16 and INT4
  KIVI (+0.196) is small in absolute terms but real.
- **`ncu` profile to verify the per-group K scale story.** Locked-clock
  Nsight Compute on INT8 vs INT4 KIVI would show the exact instruction
  counts and stall reasons that our analysis attributes the 1.29× win
  to. Same "Cause column pending" gap as Phase 1.
- **INT4 V per-token vs per-channel ablation.** We took KIVI at face
  value (per-token V) — measuring per-channel V at INT4 would
  experimentally validate the "V has no per-channel outliers" claim
  the way 2d validated the K claim.
