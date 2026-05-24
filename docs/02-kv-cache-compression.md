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

The full narrative — theory, design choice, measured result, and lesson
learned for each sub-step (2a reference → 2b INT8 → 2c INT4 KIVI →
2d perplexity), including the three surprises that overturned a
prediction one step earlier — lives in
[`02-kv-cache-compression-journey.md`](02-kv-cache-compression-journey.md).
The per-step table lives in [`results/RESULTS.md`](results/RESULTS.md).

**Headline state on `main`:** both threshold AND target met.

| Mode                                       | Memory       | Latency vs v3 | Δppl WikiText-2 | Target | Verdict |
|--------------------------------------------|-------------:|--------------:|----------------:|-------:|---------|
| **INT8 per-token K, V**                    | 65 MiB · 0.51× | tied (0.71 ms) | +0.0008         | < 0.2  | **PASS** |
| **INT4 KIVI** (per-channel K, per-token V) | **34.5 MiB · 0.27×** | **1.29× faster (0.554 ms)** | **+0.196** | **< 0.5** | **PASS** |

INT8 is essentially lossless. INT4 KIVI is the headline — smaller AND
faster AND within the quality budget. At the same INT4 bit depth, per-channel K
is **2.36× better in Δppl than naive per-token K** (0.196 vs 0.462) — direct
experimental confirmation that K's persistent outliers need their own scales.

Key lessons from this phase (each tied to a sub-step in the journey doc):

1. **"Memory-bound" is a workload property, not a kernel property** —
   INT8 halved KV bytes and tied latency with v3. On this workload the
   decode kernel was nowhere near HBM peak; the per-`j` dependency
   chain was the ceiling. Same lesson as Phase 1's v4 and v5.
2. **Scale loop nesting decides whether quantization speeds up the
   kernel** — INT8 loaded a scale per `j` and tied; INT4 KIVI loads
   scales per *group* and beats v3 by 1.29×. Moving the load out of
   the inner loop is the structural win, not the byte count.
3. **Per-channel quantization at 4 bits is friendlier to the kernel
   than per-token at 8 bits** — counter-intuitive but right, because
   per-channel scales fold cleanly into a once-per-group pre-scale of q.
4. **Kernel-level i.i.d. error is a worst-case smoke test, not a model
   quality verdict** — INT4 KIVI's 42% mean rel err on random gaussians
   shrank to a 2.78% Δppl on real activations. Real LLM activations
   have structure; softmax max-subtraction is forgiving.
5. **The KIVI insight is two-for-one** — per-channel K is the *quality*
   win (Δppl) and per-group K scale loads are the *latency* win
   (1.29× over v3). The same structural trick attacks both axes.

**Not measured (deferred to Phase 4):** decode tokens/sec at the model
level. The kernel-level bench shows the steady-state attention call is
1.29× faster with INT4 KV than fp16 KV; threading our
`decode_attention_int4` into Llama's actual KV-cache decode loop is
Phase 4 integration work.
