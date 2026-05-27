# llm-inference-kernels

Custom CUDA kernels for LLM inference, profiled and benchmarked against PyTorch,
FlashAttention, and vLLM on **Llama 3 8B**.

A capstone-grade study of the three kernels that dominate the cost of serving a
chat LLM:

| Track | Kernel | Bottleneck it attacks |
|------|--------|-----------------------|
| 1 | Fused attention (FlashAttention-style decode + prefill) | HBM traffic of the N×N attention score matrix |
| 2 | KV-cache compression (INT8 / INT4 + fused dequant) | KV-cache memory capacity & bandwidth |
| 3 | Quantized matmul (W4A16 weight-only GEMM) | Weight-memory bandwidth during decode |

## Method

Every kernel goes through one disciplined pipeline:

```
PyTorch reference  ->  naive CUDA  ->  optimized CUDA (iterative)  ->  vendor / SOTA baseline
  (correctness)        (CUDA vs CUDA: step 1)   (CUDA vs CUDA: step N)    (CUDA vs Python / SOTA)
```

- **Correctness gate** — no performance number is recorded until the kernel
  matches the PyTorch reference within a documented tolerance.
- **Incremental log** — every optimization step records before/after numbers in
  [`docs/results/RESULTS.md`](docs/results/RESULTS.md). That log *is* the
  interview deliverable.
- **Profiler-backed** — each claim is tied to an Nsight Compute metric (achieved
  bandwidth, occupancy, warp-stall reasons), not just wall-clock time.

## Comparison taxonomy

This project deliberately compares implementations along three axes:

- **CUDA vs CUDA** — naive CUDA kernel vs each successively optimized CUDA
  kernel, on the same NVIDIA GPU. This is where the *learning* lives: every
  speedup is attributed to a specific architectural cause.
- **CUDA vs Python** — the optimized CUDA kernel vs PyTorch eager and vs
  production stacks (vLLM, FlashAttention, cuBLAS).
- **C++/HIP vs CUDA** — *stretch goal.* A HIP port for AMD GPUs, kept as a
  documented portability study. See `docs/00-project-overview.md`.

## Results (filled in incrementally)

Headline numbers on RTX 4090 (sm_89). Per-step breakdown lives in
[`docs/results/RESULTS.md`](docs/results/RESULTS.md); the full v0→v5
narrative for the attention kernel is in
[`docs/01-fused-attention-journey.md`](docs/01-fused-attention-journey.md);
Phase 2 KV-quantization findings are in
[`docs/02-kv-cache-compression-journey.md`](docs/02-kv-cache-compression-journey.md);
Phase 3 W4A16 GEMM findings are in
[`docs/03-quantized-matmul-journey.md`](docs/03-quantized-matmul-journey.md).

| Kernel | Workload | Baseline | This repo | Speedup / saving | Notes |
|--------|----------|----------|-----------|------------------|-------|
| **Fused attention (decode)** | Llama 3 8B heads (`n_heads=32, n_kv_heads=8, head_dim=128`), `batch=8, seqlen_kv=4096`, fp16 | PyTorch SDPA 1.36 ms (dispatches to FlashAttention / cuDNN) | **v3: 0.713 ms, 189 GB/s achieved KV BW** | **1.91× over SDPA · 5.28× over PyTorch eager · 2.34× over our v0 baseline** | Phase 1 done; max abs diff vs fp32 reference = 3.1e-5 |
| Phase 0 end-to-end (Llama 3.1 8B Instruct, batch 16, prompt 512 / gen 512) | greedy decode, EOS suppressed | HF `generate()` 23.10 s · vLLM 0.6.6 11.65 s | n/a (vendor-baseline phase) | vLLM 1.98× HF `generate()` | Phase 0 baselines, `bench_e2e.py` |
| **KV cache, INT8 per-token** + fused dequant | same as fused attention row | fp16 KV: 128 MiB / 0.71 ms | INT8 KV: **65 MiB / 0.71 ms** | **0.51× memory** (63 MiB saved) · latency tied with v3 · Δppl **+0.0008** on WikiText-2 | Phase 2b done; essentially lossless drop-in replacement (Δppl well under 0.2 threshold) |
| **KV cache, INT4 KIVI** (per-channel K, per-token V) + fused dequant | same | fp16 KV: 128 MiB / 0.71 ms | INT4 KV: **34.5 MiB / 0.554 ms** | **0.27× memory** (93 MiB saved) · **1.29× latency** over v3 · Δppl **+0.196** on WikiText-2 | Phase 2c/2d done; clears the < 0.5 Δppl target. KIVI's per-channel K is 2.36× better than naive per-token K at the same INT4 (0.196 vs 0.462) — direct confirmation that K's persistent outliers need their own scales |
| **Quantized matmul (W4A16)** — attn QKV/O (K=4096, N=4096, M=1) | fp16 W: 32 MiB | fp16 cuBLAS 0.047 ms | INT4 W: **8.25 MiB / 0.016 ms** | **0.26× memory** · **2.88× latency** | Phase 3c; symmetric INT4 per-channel groupwise (group=128) |
| **Quantized matmul (W4A16)** — MLP up/gate (K=4096, N=14336, M=1) | fp16 W: 112 MiB | fp16 cuBLAS 0.134 ms | INT4 W: **28.88 MiB / 0.019 ms** | **0.26× memory** · **6.97× latency** | Phase 3c; the headline shape — 7× over cuBLAS |
| Quantized matmul (W4A16) — MLP down (K=14336, N=4096, M=1) | fp16 W: 112 MiB | fp16 cuBLAS 0.133 ms | INT4 W: 28.88 MiB / **0.045 ms** | **0.26× memory** · **2.96× latency** | Phase 3c |
| **E2E Phase 4a** — Llama 3.1 8B + fused attention | locked workload (batch=16, prompt=512, gen=512), greedy | vanilla HF 335.8 tok/s · 18.50 GB peak VRAM · MMLU 68.32% | **344.1 tok/s · 18.57 GB · MMLU 68.32%** | **+2.5% tok/s · bit-identical accuracy** | Phase 4a; attention bit-perfect (greedy_match=1.0). Small e2e gain because attention is ~4% of decode time at this workload |
| **E2E Phase 4b** — + INT4 KIVI KV cache | same | vanilla as above | **521.7 tok/s · 18.41 GB · MMLU 67.29%** | **1.55× tok/s · −1.03 pp MMLU · Δppl +0.20** | Phase 4b; PPL delta matches Phase 2c's kernel-level number to within rounding; real INT4 cache class via HF Cache subclass |
| **E2E Phase 4c** — + W4A16 weights (memory headline) | same | vanilla as above | **199.9 tok/s (B=16, w/ Phase 5 kernel)** · 56.9 tok/s (B=1) · **9.05 GB peak VRAM** · MMLU 62.40% | **−51% peak VRAM (-9.45 GB) · 0.60× vs vanilla at B=16** · 1.16× tok/s at B=1 · −5.92 pp MMLU | Phase 4c + Phase 5; 4c initially regressed to 0.12× at B=16 because the Phase 3 W4A16 kernel was M=1-only. Phase 5 adds a batched-decode kernel (BLOCK_M=16, K-split across warps) that amortizes weight bandwidth across the batch — 4.9× recovery. Residual gap to cuBLAS is the tensor-core advantage (scalar fp32 vs fp16 MMA); closing it is a Phase 6 follow-up |

## Repo layout

```
docs/         Project overview, per-track design docs, methodology, results log
kernels/      CUDA kernels: attention/, kv_cache/, quant/, common/
bindings/     Python <-> CUDA glue (PyTorch C++ extension)
reference/    PyTorch reference implementations — the correctness oracles
benchmarks/   Timing harness + per-kernel and end-to-end benchmarks
tests/        Correctness tests (kernel output vs reference)
scripts/      Environment detection, setup
CMakeLists.txt  Standalone CUDA microbenchmark build
setup.py        PyTorch extension build
```

## Quickstart

```bash
# 1. reproducible conda environment (Python 3.11, CUDA 12.4 toolkit)
conda env create -f environment.yml
conda activate llm-inference-kernels
pip install flash-attn --no-build-isolation
bash scripts/detect_env.sh                 # writes docs/results/env-report.md

# 2. build the PyTorch extension (once kernels exist)
python setup.py build_ext --inplace

# 3. standalone CUDA microbenchmarks
cmake -B build -S . && cmake --build build

# 4. run benchmarks
python benchmarks/bench_attention.py
```

**For a complete walkthrough** — what to run for each phase, what numbers
to expect, what to look for in each kernel — see
[`docs/running-the-repo.md`](docs/running-the-repo.md). It covers the
`v0`–`v5` Phase 1 attention branches, the `2a`–`2d` Phase 2 KV-cache
branches, expected headline numbers per branch, and the common gotchas
(rebuild after checkout, conda activation, etc.).

## Status

See [`TODO.md`](TODO.md) for the phased, step-by-step plan.

**Phase 0 — environment & baselines: done.** Reproducible conda env locked
(`environment.lock.yml`), Llama 3.1 8B Instruct verified loading + generating
at fp16, vLLM + HF `generate()` baselines captured in
[`docs/results/RESULTS.md`](docs/results/RESULTS.md).

**Phase 1 — fused decode attention: substantially complete.** Six kernel
versions explored (v0 → v5); **v3 lives on `main` at 0.713 ms / 189 GB/s, 1.91×
faster than PyTorch SDPA** on the reference microbench workload. v4
(FlashDecoding split-K) and v5 (`cp.async` double-buffering) were explored
but regressed on this workload — both retained in git history with diagnostic
writeups in
[`docs/01-fused-attention-journey.md`](docs/01-fused-attention-journey.md).
Remaining: direct comparison vs raw `flash_attn` (currently we have it
indirectly via SDPA), `ncu` profile with locked clocks for the "Cause"
column in RESULTS.md, stretch goals (tensor-core MMA path, prefill FA-2
forward kernel).

**Phase 2 — KV-cache compression: complete.** INT8 per-token KV is
essentially lossless (Δppl +0.0008, 0.51× memory, latency tied with v3).
**INT4 KIVI** (per-channel K + per-token V, packed 4-bit) clears the
< 0.5 Δppl target on WikiText-2 with margin (**Δppl +0.196**) at
**0.27× memory** and **1.29× faster** than the fp16 KV path. Both
docs/02 threshold AND target met. The KIVI structural change — K scales
load once per group rather than per j, getting them out of the
inner-loop dependency chain — is what made INT4 *also* a latency win,
not just a memory one. Full findings in
[`docs/02-kv-cache-compression.md`](docs/02-kv-cache-compression.md).
Decode-tok/s at the model level deferred to Phase 4 (requires plumbing
the INT4 attention kernel into Llama's actual KV-cache decode loop).

**Phase 3 — quantized matmul (W4A16): complete.** Symmetric INT4 weight
quantization with groupwise scales (group=128 along K). The naive 3b
kernel already beat fp16 cuBLAS by 1.59× on Llama 3 8B's MLP up/gate
shape (the headline decode shape). The 3c optimization — multi-warp
blocks with K split across warps + `act` cached in shared memory —
pushed the wins to **2.88× / 6.97× / 2.96× over fp16 cuBLAS** on the
three M=1 layer shapes (attn QKV/O, MLP up/gate, MLP down). All clear
the docs/03 *Target* of 2–3× speedup. Full findings in
[`docs/03-quantized-matmul-journey.md`](docs/03-quantized-matmul-journey.md).
Marlin head-to-head and GPTQ perplexity validation deferred.

**Phase 4 — end-to-end integration on Llama 3.1 8B Instruct: complete.**
Three monkeypatches landed one-kernel-at-a-time, with full
MMLU/HellaSwag/ARC-C + WikiText-2 PPL + tokens/sec + peak VRAM at each
step. Headlines on the locked workload (batch=16, prompt=512, gen=512):
**4a** fused attention is bit-identical to vanilla on every accuracy
metric and +2.5% tok/s; **4b** INT4 KIVI KV cache hits 1.55× tok/s for
-1.03 pp MMLU + Δppl +0.20 (matches Phase 2c kernel-level prediction);
**4c** W4A16 weights cut peak VRAM 51% (18.50 → 9.05 GB) and give
1.16× tok/s at batch=1, but regress to 0.12× at batch=16 because the
Phase 3 kernel is M=1-only and the locked workload has M=16 — the
batched-decode kernel is Phase 5. Full narrative in
[`docs/04-end-to-end-integration-journey.md`](docs/04-end-to-end-integration-journey.md).

**Phase 5 — batched-decode W4A16 kernel: complete.** Added
`w4a16_gemm_batched_decode_kernel` for `M ∈ [2, 16]` — same K-split-
across-warps pattern as Phase 3c but each thread now accumulates a
length-`BLOCK_M=16` vector of fp32 partials, and each warp has its own
`[BLOCK_M, group_size]` activation tile in shmem so the int4 weight
read is amortized across all M rows. **Phase 4c at batch=16 recovers
from 40.9 to 199.9 tok/s (4.9× over the M=1-only baseline; 0.60× vs
vanilla HF).** Residual gap is the tensor-core advantage cuBLAS holds
at this M; an `mma.sync`-based rewrite is the natural Phase 6.
