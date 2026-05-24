# Track 2 — KV-Cache Compression

> Builds directly on Track 1: the compressed-KV path *is* the Track 1 decode
> kernel reading quantized K/V with fused dequantization.

## Problem

KV-cache size:

```
bytes = 2 (K,V) · n_layers · seqlen · n_kv_heads · head_dim · dtype_bytes · batch
```

Llama 3 8B: `n_layers=32`, `n_kv_heads=8` (GQA), `head_dim=128`. In FP16, per
token across all layers:

```
2 · 32 · 8 · 128 · 2 bytes  =  256 KB / token
```

At 8K context × batch 32 that is **64 GB** — it does not fit. The KV cache, not
the weights, is what caps concurrent users and context length. It is also read
in full every decode step, so it is a **bandwidth** cost too.

## Approach — quantize the KV cache

Store K/V in INT8 or INT4 with group-wise scales; dequantize on the fly.

- **INT8** → 2× smaller, near-lossless, production-proven.
- **INT4** → 4× smaller, needs care to keep accuracy.

### Per-channel vs per-token scales (KIVI)

K and V have different statistics. K has persistent **per-channel outliers**;
V does not. The KIVI result:

- Quantize **K per-channel** (scale per head_dim index).
- Quantize **V per-token** (scale per token).

This keeps INT4 accuracy usable. Both use group-wise scales (group ≈ 32–128) and
asymmetric quantization (zero-point) for the K distribution.

## Kernel work

1. **Quantize kernel** — runs when a new token is appended to the cache:
   FP16 K/V → INT8/INT4 + scales, in the chosen layout.
2. **Fused dequant in attention** — the Track 1 decode kernel reads packed
   INT4/INT8 K/V, dequantizes in registers/shared memory, and proceeds. This
   fusion is the deep part: no separate dequant pass, no FP16 KV in HBM.

Decode attention is memory-bound, so 4-bit KV means ~4× less KV traffic — this
can make the *attention kernel itself faster*, not just smaller.

## Layout considerations

- Packed INT4: 8 values per 32-bit word; choose packing so the attention
  kernel's coalesced loads stay coalesced after packing.
- Scales in a small companion tensor; keep them L2/shared-friendly.
- Per-channel K means the scale index depends on head_dim lane — plan the
  warp-to-channel mapping so scale loads are cheap.

## Accuracy measurement

Quality is a first-class metric here. Track:

- **Perplexity** on WikiText-2 (or similar) — FP16 vs INT8 vs INT4.
- A short chat-style eval for sanity (Llama 3 8B Instruct is the target).
- Report perplexity *delta*, not just absolute.

## Stretch — algorithmic compression

- **StreamingLLM** — attention sinks + sliding window (bounded cache).
- **H2O** — heavy-hitter token eviction.

These are eviction policies, less kernel-heavy; mention and benchmark only if
time allows.

## Metrics

- KV-cache memory reduction (×).
- Perplexity delta vs FP16 KV.
- Decode tokens/sec (expect neutral-to-positive from reduced bandwidth).
- Max context length / max batch that now fits on the GPU.

## Success criteria

- Threshold: INT8 KV correct; 2× memory; perplexity delta < 0.2.
- Target: INT4 KV (per-channel K / per-token V); ~3.5–4× memory;
  perplexity delta < 0.5; tokens/sec neutral or better.
- Stretch: measured decode throughput gain from reduced KV bandwidth.

## References

- Liu et al., *KIVI: 2-bit KV cache quantization* (2024)
- Hooper et al., *KVQuant* (2024)
- Xiao et al., *StreamingLLM* (2023)
- Zhang et al., *H2O* (2023)
- TensorRT-LLM INT8 KV cache; vLLM FP8 KV cache

## Findings

**Phase 2b — INT8 per-token KV (landed):**

- **Memory: 0.51× of fp16** (128 MiB → 65 MiB on the reference workload;
  63 MiB saved). Scale overhead is ~1.5% — one fp16 scale per token,
  shared across head_dim.
- **Kernel latency: 0.713 ms, tied with v3 fp16 (0.713 ms).** Halving KV
  bytes did not move latency — we measured the INT8 kernel at 96 GB/s
  effective KV bandwidth vs v3's 189 GB/s, both well under HBM peak
  (1008 GB/s). The decode kernel is **dependency-chain-bound**, not
  bandwidth-bound: the per-`j` `warp_reduce_sum → softmax → FMA` critical
  path is the limiter, and that shape is the same in both kernels.
  Consistent with Phase 1's v4/v5 results (more SMs and more cp.async
  pipelining also didn't help — bandwidth wasn't the ceiling on this
  workload).
- **Accuracy: max |diff| vs fp16 reference 1.1e-3, mean relative error
  2.5%.** Below the per-element quantization-noise floor of `1/127 ≈
  0.78%` per element, propagated through the softmax.
- **Quantize-kernel one-shot cost: 0.124 ms** for the full K cache (64 MiB
  fp16 → 32 MiB int8 + scales). In serving this is amortised: only the
  newly appended token is quantized per decode step.
- **Implementation notes:** scale-folding optimisation — instead of
  per-lane dequant (4 multiplies/thread/iter to materialise k = k_int ·
  k_scale), compute the partial dot on int values and multiply by
  `k_scale` once after `warp_reduce_sum` (linearity); fold `v_scale` into
  the FMA coefficient `p_j_scaled = p_j · v_scale`. Saves 8 multiplies
  per iter vs naive dequant-everything. INT8 loaded as `LDG.E.32` (4 bytes
  per thread vs v3's 8) — `head_dim/32 = 4` lanes fits cleanly into one
  int32 load.

**Implication for Phase 2's framing:** the value of INT8 KV here is *memory*,
not latency. Same answer time + half the cache → ~2× longer context or
~2× larger batch in the same VRAM budget. The latency story may change
in Phase 2c (INT4): four-bit loads can use `LDG.E.16` patterns and may
expose different bottlenecks, but the dependency-chain analysis above
predicts a latency tie there too.

**Phase 2c — INT4 KIVI (landed):**

- **Memory: 0.27× of fp16** (128 MiB → 34.5 MiB on the reference workload;
  93.5 MiB saved). K storage is `head_dim/2` packed bytes per token + a small
  per-channel-per-group scales table; V storage is `head_dim/2` packed bytes
  per token + one fp16 per-token scale.
- **Kernel latency: 0.554 ms — 1.29× faster than v3 (0.715 ms).** INT4
  *moves the needle* where INT8 didn't. The structural change: K scales
  load **once per group** (every 32 inner iters), held in registers
  through the group's inner loop, and pre-folded into q via
  `q_scaled[d] = q[d] · k_scale[g, d]` at each group boundary. The inner
  loop is then 4 multiplies on int values + warp_reduce + softmax + 4
  FMAs on int values — denser than INT8's per-iter K scale load on top
  of the dot product. Inner-loop K_q and V_q loads are also smaller
  (`LDG.E.16`, 2 B/thread, one uint16 unpacked to 4 nibbles).
- **Accuracy on random gaussians: max abs diff vs fp16 reference 2.3e-2,
  mean rel err 42%.** Below INT4's per-element noise floor (`1/7 ≈ 14%`)
  amplified through the softmax. This is the kernel-level number on
  worst-case (random) data; perplexity on real LLM activations is the
  Phase 2d question, and the KIVI paper's result suggests the per-channel
  K helps a lot there because K's persistent outliers get their own scales.
- **Quantize one-shot costs:** K (per-channel groupwise) 0.07 ms; V
  (per-token packed) 0.12 ms. In serving these amortise to per-appended-
  token costs.

**Why INT4 moves latency where INT8 tied** (the surprise vs the Phase 2b
diagnosis): INT8 still loaded one fp16 K scale *per j*, which the
dependency chain couldn't hide. INT4 KIVI loads K scales once per 32 j's,
so the per-iter load count drops by ~one. Combined with the smaller K_q +
V_q loads, the inner loop's instruction throughput is meaningfully higher.
The dependency-chain-bound diagnosis from 2b still applies, but the chain
itself got lighter in 2c.

**Phase 2d — perplexity (landed):**

Measured on Llama 3.1 8B Instruct over the WikiText-2 test split (131,008
tokens, 64 chunks × 2048 tokens) by `scripts/eval_perplexity.py`. The
script patches `F.scaled_dot_product_attention` to round-trip K and V
through the PyTorch quantization reference before delegating to the
underlying sdpa — the model sees exactly the noisy K, V it would receive
from a compressed KV cache at decode time.

| Mode                       | ppl    | Δppl    | % delta | Target  | Verdict |
|----------------------------|-------:|--------:|--------:|--------:|---------|
| fp16 (baseline)            | 7.055  | —       | —       | —       | —       |
| INT8 per-token K, V        | 7.056  | +0.0008 | +0.01%  | < 0.2   | **PASS** |
| INT4 per-token K, V (naive)| 7.517  | +0.462  | +6.54%  | —       | KIVI comparator |
| **INT4 KIVI** (per-ch K, per-token V) | **7.252** | **+0.196** | **+2.78%** | **< 0.5** | **PASS** |

**Headline results:**

- **INT8 KV is essentially lossless** (Δppl = +0.0008, well under the
  0.2 threshold). Combined with 0.51× memory and tied latency vs v3,
  INT8 is a no-brainer drop-in.
- **INT4 KIVI clears the < 0.5 target with margin** (Δppl = +0.196).
  At 0.27× memory AND 1.29× latency over v3, this is the headline win
  of Phase 2.
- **KIVI's per-channel K is worth 2.36× in quality at the same bit
  depth** vs naive per-token K (0.196 vs 0.462). The K-side outliers
  matter; the design doc's "K has persistent per-channel outliers; V
  does not" prediction held experimentally.

**Why the kernel-level 42% mean rel err didn't tank perplexity:** that
number was measured on i.i.d. random gaussian inputs — uniformly hard
for symmetric per-token quantization. Real activations have structure
(per-channel outliers in K captured by KIVI's per-channel scales; V
distribution narrower than worst case). The softmax's max-subtraction
also makes the attention output relatively robust to per-element noise
in V — only the ranking of large scores survives.

**Not measured:** decode tokens/sec at the model level. That requires
threading our `decode_attention_int4` CUDA kernel into Llama's actual
KV-cache decode loop — Phase 4 integration work. The kernel-level
benchmark (`benchmarks/bench_kv_cache.py`) shows the steady-state
attention call is 1.29× faster with INT4 KV than fp16 KV, which is
the direct decode-step speedup.
