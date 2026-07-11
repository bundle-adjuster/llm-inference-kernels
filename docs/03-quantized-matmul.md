# Track 3 — Quantized Matmul (W4A16 GEMM)

> The hardest track. Scope honestly: the achievable, defensible win is the
> **decode-shape, memory-bound GEMM**, not beating cuBLAS on large GEMMs.

## Problem

Llama 3 8B has ~7B of its 8B parameters in linear layers (QKV/O projections,
MLP up/gate/down). In FP16 that is ~14 GB of weights.

In **decode**, each linear layer is `[M] × [K×N]` with `M = batch · 1` (tiny).
The GEMM is **memory-bound on weight traffic** — the runtime is dominated by
reading the weight matrix from HBM, not by the FLOPs.

So: **weight-only quantization (W4A16)** — store weights in INT4 (~3.5 GB),
keep activations FP16. ~4× less weight HBM traffic → up to ~4× faster decode
GEMM. This is the GPTQ/AWQ family; the SOTA fast kernel is **Marlin**.

## Why W4A16 (not W8A8) is primary

- **W4A16** — weight-only. No activation quantization, no calibration of
  activations, accuracy is well understood (GPTQ/AWQ). Wins exactly the
  memory-bound decode regime. **This is the main kernel.**
- **W8A8** — INT8 weights *and* activations; uses INT8 Tensor Cores; needs
  activation quantization (SmoothQuant). Helps compute-bound prefill. Optional
  variant.

## Kernel design — W4A16 GEMM

- Weights stored INT4, packed 8 per 32-bit word, with **group-wise scales**
  (group size 128 along K) and optional zero-points.
- Per output tile: load packed INT4 weights, **dequantize in registers**
  (`w_fp16 = (q - z) · scale`), multiply by FP16 activations, accumulate.
- Decode (`M` tiny): memory-bound — focus on weight load efficiency, vectorized
  INT4 unpacking, minimal redundant scale loads. CUDA cores are fine here.
- Prefill (`M` large): compute-bound — dequantize into shared memory, feed
  **FP16 Tensor Core MMA** (`mma.sync`). This is where Marlin's tricks matter.

## Hardware notes

- INT4 *storage* + register dequant works on any Ampere/Ada GPU.
- **FP8** Tensor Core path: gated on **sm_89+ (Ada/Hopper)** — Ampere lacks FP8.
  `scripts/detect_env.sh` records the capability; build flags key off it.
- INT8 Tensor Cores (for the W8A8 variant) exist on Ampere+.

## Baselines (CUDA vs Python / SOTA)

- FP16 cuBLAS (`torch.matmul`) — the bar to beat on decode shapes.
- Marlin — SOTA W4A16 kernel; the "how close to SOTA" reference.
- bitsandbytes / AWQ kernels — secondary references.

## Test shapes

Use real Llama 3 8B layer shapes:

- MLP: `K=4096, N=14336` (up/gate), `K=14336, N=4096` (down).
- Attention: `K=4096, N=4096` (QKV fused / O).
- Sweep `M ∈ {1, 8, 32, 128, 512}` to cross the memory-bound → compute-bound
  boundary and show *where* the quantized kernel wins.

## Metrics

- Latency vs FP16 cuBLAS, per shape, across the `M` sweep.
- Achieved HBM bandwidth (decode) / achieved TFLOP/s (prefill).
- Effective weight footprint reduction.
- Accuracy: perplexity with W4A16 weights (GPTQ/AWQ-quantized).

## Success criteria

- Threshold: correct vs reference; beats FP16 cuBLAS on decode-shape GEMM.
- Target: 2–3× over FP16 cuBLAS on decode shapes; within 25% of Marlin.
- Stretch: competitive on prefill (compute-bound) shapes.

## References

- Frantar et al., *GPTQ* (2022)
- Lin et al., *AWQ* (2023)
- Frantar et al., *Marlin* (2024)
- Xiao et al., *SmoothQuant* (2022)
- Dettmers et al., *LLM.int8()* (2022)
- NVIDIA CUTLASS — mixed-input GEMM

## Findings

The full narrative — theory, design choice, measured result, lesson per
sub-step (3a reference → 3b naive → 3c decode-optimised → 3d wrap) —
lives in [`03-quantized-matmul-journey.md`](03-quantized-matmul-journey.md).
The per-step quantitative table lives in
[`results/RESULTS.md`](results/RESULTS.md).

**Headline state on `main`:** Phase 3 *Target* hit on all three Llama
3 8B M=1 layer shapes.

| Shape (K, N)    | M=1 cuBLAS | **3c decode** | Speedup |
|-----------------|-----------:|--------------:|--------:|
| 4096 × 4096 (attn QKV / O)    | 0.047 ms | **0.016 ms** | **2.88×** |
| **4096 × 14336 (MLP up/gate)**| 0.134 ms | **0.019 ms** | **6.97×** |
| 14336 × 4096 (MLP down)        | 0.133 ms | **0.045 ms** | **2.96×** |

The 3b → 3c jump (2.88×–6.97×, depending on shape) came from two
combined changes:

1. **Multi-warp block + K split across warps** — 4 warps per block
   (vs 1 in 3b), with K split 4-way across them and a tiny shmem
   combine at the end. Addresses both the "long sequential K" loss
   (mlp-down) and the "SM under-occupation at small N" loss
   (attn-square) in one structural change.
2. **`act` in shared memory** — cooperative load once per block,
   frees L1 for weight traffic.

Both changes are pure additions (no trade-offs against other shapes),
which is why the improvement over 3b is so uniform across shapes
(5.4× / 4.4× / 6.3×).

Key lessons from this phase (each tied to a sub-step in the journey doc):

1. **"Memory-bound" thesis carries across kernel families.** The
   W4A16 win on Llama linear layers is the same pattern as Phase 1
   attention's "find the workload where bandwidth is the ceiling and
   exploit smaller bytes."
2. **K-split across warps pays once the block is multi-warp.** Phase 1
   v4's split-K was blamed on attention's per-iter dependency chain /
   "bandwidth was the ceiling" — that diagnosis was wrong (Phase 7). The
   real problem was occupancy: v4's split-K sat on a **single-warp** block
   that filled ~2 of 128 SMs. Phase 8's v6 puts the same split-K on
   **multi-warp** blocks (4 warps) and it converts cleanly — beats fair
   GQA-native SDPA (1.01×, ~82% of peak HBM, 4.59× over v3). 3c's GEMM
   K-split worked immediately for the same reason: it was already
   multi-warp, so it never hit the occupancy wall attention did. See
   [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).
3. **Multi-warp blocks pay when there's K-side compute to split.**
   3c's K-split + 3b's M-loop are the same "move work out of the
   inner serial loop" idea applied to different axes.
4. **Shmem-cache the data that's reused per-block.** L1 was already
   catching `act` reuse across the warp; explicit shmem caching
   freed L1 for the new bottleneck (weight reads).
5. **L2 effectiveness is real.** Warm-cache W4A16 hits 1577 GB/s
   "effective bandwidth" on MLP up/gate because 28.88 MiB packed
   weights fit in the 72 MB L2. Realistic decode regime.

**Not measured (deferred):**

- Marlin head-to-head (docs/03 *Target* "within 25% of Marlin").
- GPTQ / AWQ quantisation + perplexity on real Llama 3 8B weights.
- Tensor Core MMA path for prefill (M ≥ 16) — stretch.
- M > 1 fast path (M-fast inner loop). Currently M > 1 falls back to
  the 3b naive kernel via the launcher; the proper batched-decode
  variant is a Phase 4 integration concern.
