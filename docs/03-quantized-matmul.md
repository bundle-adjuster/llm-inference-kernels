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

## Findings (fill in as you go)

_Decode shape `M=1`: ___× over cuBLAS, ___% of Marlin._
_M-sweep crossover point: ___._
