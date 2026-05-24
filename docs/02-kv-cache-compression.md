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

**Phase 2c — INT4 KIVI (per-channel K, per-token V): pending.**
**Phase 2d — perplexity / decode tok/s: pending.**
