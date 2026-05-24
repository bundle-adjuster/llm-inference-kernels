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
[`docs/02-kv-cache-compression.md`](docs/02-kv-cache-compression.md).

| Kernel | Workload | Baseline | This repo | Speedup / saving | Notes |
|--------|----------|----------|-----------|------------------|-------|
| **Fused attention (decode)** | Llama 3 8B heads (`n_heads=32, n_kv_heads=8, head_dim=128`), `batch=8, seqlen_kv=4096`, fp16 | PyTorch SDPA 1.36 ms (dispatches to FlashAttention / cuDNN) | **v3: 0.713 ms, 189 GB/s achieved KV BW** | **1.91× over SDPA · 5.28× over PyTorch eager · 2.34× over our v0 baseline** | Phase 1 done; max abs diff vs fp32 reference = 3.1e-5 |
| Phase 0 end-to-end (Llama 3.1 8B Instruct, batch 16, prompt 512 / gen 512) | greedy decode, EOS suppressed | HF `generate()` 23.10 s · vLLM 0.6.6 11.65 s | n/a (vendor-baseline phase) | vLLM 1.98× HF `generate()` | Phase 0 baselines, `bench_e2e.py` |
| **KV cache, INT8 per-token** + fused dequant | same as fused attention row | fp16 KV: 128 MiB / 0.71 ms | INT8 KV: **65 MiB / 0.71 ms** | **0.51× memory** (63 MiB saved) · latency tied with v3 · Δppl **+0.0008** on WikiText-2 | Phase 2b done; essentially lossless drop-in replacement (Δppl well under 0.2 threshold) |
| **KV cache, INT4 KIVI** (per-channel K, per-token V) + fused dequant | same | fp16 KV: 128 MiB / 0.71 ms | INT4 KV: **34.5 MiB / 0.554 ms** | **0.27× memory** (93 MiB saved) · **1.29× latency** over v3 · Δppl **+0.196** on WikiText-2 | Phase 2c/2d done; clears the < 0.5 Δppl target. KIVI's per-channel K is 2.36× better than naive per-token K at the same INT4 (0.196 vs 0.462) — direct confirmation that K's persistent outliers need their own scales |
| Quantized matmul (W4A16) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | Phase 3 (next) |
| End-to-end (Llama 3 8B, kernels integrated) | _TBD_ | vanilla vLLM 703 tok/s | _TBD_ | _TBD_ | Phase 4 |

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

**Phase 3 — quantized matmul (W4A16): next, unblocked.**
