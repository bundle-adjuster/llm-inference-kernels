# Project Overview

## One-line goal

Hand-write CUDA kernels for the three hotspots of LLM inference — attention,
the KV cache, and the linear-layer matmuls — take each from a naive
implementation to an optimized one, and produce a rigorous, profiler-backed
account of *where every speedup comes from*, benchmarked end-to-end against
vanilla vLLM on Llama 3 8B.

## Why this project

Serving a chat LLM is dominated by three costs:

1. **Attention** reads and writes the KV cache every decode step; a naive
   implementation also materialises the N×N score matrix in HBM.
2. **The KV cache itself** is the memory-capacity wall — it grows linearly with
   context length and batch size and decides how many users fit on a GPU.
3. **The linear layers** (QKV/O projections, MLP) are ~7B of the 8B parameters;
   in decode they are memory-bound on weight traffic.

These are exactly the problems a production inference team works on. The point
of the project is not to beat vLLM outright (years of tuning back it) — it is to
*understand it from the metal up* and to be able to defend, with profiler
evidence, every microsecond.

## Scope

| In scope | Out of scope (this round) |
|----------|---------------------------|
| Llama 3 8B as the single target model | Multi-GPU / tensor parallelism |
| Single NVIDIA GPU (Ampere/Ada), FP16 base | Training, fine-tuning, LoRA |
| Decode-path focus (prefill as secondary) | Speculative decoding, MoE |
| Fused attention, KV compression, W4A16 GEMM | Custom scheduler / continuous batching |
| HIP/AMD port as a *documented stretch goal* | A production-ready serving system |

## The three tracks (built in this order)

1. **Fused attention** — FlashAttention-style streaming-softmax attention.
   Primary target: the **decode** kernel (single query token vs a long KV
   cache), which is memory-bound and the dominant cost in chat generation.
   Prefill (FlashAttention-2 forward) is a secondary sub-step.
2. **KV-cache compression** — quantize the KV cache to INT8/INT4 with
   group-wise scales and **fuse dequantization into the attention kernel from
   Track 1**. Wins on both memory capacity and bandwidth.
3. **Quantized matmul** — W4A16 weight-only GEMM (GPTQ/AWQ-style) for the linear
   layers. Primary target: decode-shape (tall-skinny) GEMMs where 4-bit weights
   cut HBM traffic ~4×.

Each track builds on the last; Track 2's kernel is literally Track 1's kernel
reading quantized inputs.

## Comparison methodology

For every track we produce four implementations and compare them:

| # | Implementation | Role | Axis |
|---|----------------|------|------|
| A | PyTorch reference | correctness oracle | — |
| B | Naive CUDA kernel | first working kernel | CUDA vs CUDA (baseline) |
| C | Optimized CUDA kernel(s) | iterative, one optimization at a time | CUDA vs CUDA (each step) |
| D | Vendor / SOTA (cuBLAS, FlashAttention, vLLM) | the bar | CUDA vs Python / SOTA |

The **B → C** progression is the deep-learning core: each optimization is landed
as its own commit + RESULTS.md entry, with an Nsight Compute metric explaining
*why* it helped.

## Success criteria (tiered, honest)

Targets are deliberately tiered. "Threshold" is the minimum for the track to be
considered done; "Target" is the goal; "Stretch" is a strong result.

### Track 1 — Fused attention (decode)
- **Threshold:** correct vs reference; ≥3× faster than naive PyTorch eager.
- **Target:** within 20% of `flash_attn` decode on achieved HBM bandwidth.
- **Stretch:** within 10%; working prefill FA-2 forward path.

> **Status (Phase 8):** Target met and exceeded. The v6 FlashDecoding split-K
> kernel matches GQA-native `F.scaled_dot_product_attention` on the Phase 1
> reference workload (155.6us vs 157.3us = 1.01×, ~82% of peak HBM), and beats it
> on all HBM-bound shapes (kv≥2048). It trails only on small L2-resident shapes
> (kv≤1024, 0.69–0.82×), which is the honest remaining boundary. Note: the older
> v3 kernel actually *lost* to a fair GQA-native baseline (0.22×) — the earlier
> "1.91× over SDPA" figure compared against SDPA fed a 4×-expanded GQA KV cache.
> See `docs/05-baseline-correction-journey.md` (the correction) and
> `docs/06-attention-splitk-journey.md` (the v6 fix).

### Track 2 — KV-cache compression
- **Threshold:** INT8 KV correct; 2× KV memory reduction; perplexity delta < 0.2.
- **Target:** INT4 KV (per-channel K, per-token V); ~3.5–4× memory reduction;
  perplexity delta < 0.5; tokens/sec neutral or better.
- **Stretch:** measured throughput gain from reduced KV bandwidth in decode.

### Track 3 — Quantized matmul (W4A16)
- **Threshold:** correct vs reference; beats FP16 cuBLAS on decode-shape GEMM.
- **Target:** 2–3× over FP16 cuBLAS on decode shapes; within 25% of Marlin.
- **Stretch:** competitive on prefill (compute-bound) shapes.

### End-to-end
- **Threshold:** Llama 3 8B runs correctly with all three custom kernels.
- **Target:** ≥1.3× tokens/sec over vanilla vLLM on a decode-heavy workload,
  *or* a fully profiler-backed account of why not.

> Note: an honest "within 12% of FlashAttention, and here is the exact
> profiler reason for the gap" is a stronger interview artifact than an
> unverifiable headline number. Rigor is the deliverable.

## Deliverables

- This repo: kernels, tests, benchmarks, build system.
- `docs/results/RESULTS.md` — the incremental optimization log (the headline
  artifact: every step, every number, every profiler metric).
- Per-track design docs (`docs/01..03`) written *before* coding each track.
- A final README results table + a slide-ready summary distilled from RESULTS.md.

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Beating SOTA is hard | Success criteria are tiered; "within X% + why" counts. |
| Llama 3 8B FP16 ≈ 16 GB — tight VRAM | Decode benchmarks use small batch; quantize early; rent a bigger GPU only for end-to-end if needed. |
| Tensor Core MMA is awkward for M=1 decode | Decode kernels stay on CUDA cores where appropriate; document the tradeoff rather than forcing MMA. |
| Scope creep across 3 kernels in 4–6 weeks | Track 1 must hit *Target* before Track 2 starts; Tracks 2/3 may land at *Threshold*. |
| Unfair benchmarking | `docs/benchmarking-methodology.md` fixes the protocol before any number is taken. |

## Timeline (4–6 weeks)

| Week | Focus |
|------|-------|
| 1 | Phase 0 + Phase 1 naive attention kernel correct |
| 2 | Phase 1 optimization sweep + write-up |
| 3 | Phase 2 KV-cache compression |
| 4 | Phase 3 quantized matmul |
| 5 | Phase 4 end-to-end integration vs vLLM |
| 6 | Buffer: profiling deep-dives, RESULTS.md polish, presentation |

## Hardware target

Single NVIDIA GPU, compute capability 8.0–8.9 (Ampere/Ada). `cp.async` and
INT8/INT4 tensor cores are assumed; FP8 paths are gated on sm_89+. Run
`scripts/detect_env.sh` first — it records the exact device into
`docs/results/env-report.md`, and kernel build flags key off it.

## Stretch goal — AMD / HIP portability

AMD is **not** in the main 4–6 week plan. As a documented stretch goal, the
optimized CUDA kernels can be ported to HIP (portable C++) and benchmarked on an
AMD GPU — turning the "C++ vs CUDA" idea into a real cross-vendor portability and
performance study. Tracked at the bottom of `TODO.md`.
