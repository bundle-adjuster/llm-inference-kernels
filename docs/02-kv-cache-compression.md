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

## Findings (fill in as you go)

_INT8: ___× memory, ___ ppl delta._
_INT4 (per-channel K): ___× memory, ___ ppl delta, ___ tokens/sec._
