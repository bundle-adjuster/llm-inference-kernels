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

| Kernel | Workload | Baseline | This repo | Speedup | Memory | Notes |
|--------|----------|----------|-----------|---------|--------|-------|
| Fused attention (decode) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Phase 1 |
| KV-cache compression | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Phase 2 |
| Quantized matmul (W4A16) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Phase 3 |
| End-to-end (Llama 3 8B) | _TBD_ | vanilla vLLM | _TBD_ | _TBD_ | _TBD_ | Phase 4 |

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

## Status

See [`TODO.md`](TODO.md) for the phased, step-by-step plan. Current phase:
**Phase 0 — environment & baselines.**
