# TODO — phased, step-by-step plan

Work top to bottom. Each phase has an **exit criterion** — do not start the next
phase until it is met. Check items off as you land them; each optimization step
is its own commit + a row in `docs/results/RESULTS.md`.

Legend: `[ ]` todo · `[~]` in progress / partial / explored-not-landed · `[x]` done

---

## Phase 0 — Environment, baselines, infrastructure

- [x] Create the reproducible conda env: `conda env create -f environment.yml`
      (Python 3.11, CUDA 12.4 toolkit; target GPU RTX 4090 / sm_89)
- [x] Install flash-attn: `MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation`
- [x] Confirm `torch.cuda.is_available()` and the cu124 build inside the env
- [x] After install succeeds, lock it:
      `conda env export --no-builds > environment.lock.yml`
- [x] Run `scripts/detect_env.sh` → `docs/results/env-report.md`
- [x] Download Llama 3.1 8B Instruct weights; confirms FP16 load + generation
      (`scripts/load_llama.py`): 16.06 GB weights, 16.08 GB peak VRAM
- [x] Stand up vLLM; record baseline tokens/sec on the reference workload
      (`benchmarks/bench_e2e.py`): vLLM 703 tok/s, HF generate 355 tok/s
- [x] Verify `benchmarks/harness.py` — used across v0–v5; CUDA-event timing + `check_close` exercised every step
- [x] Lock the reference workload (batch, prompt len, gen len) in
      `docs/benchmarking-methodology.md`
- [~] Lock GPU clocks; document the measurement protocol — `scripts/lock_clocks.sh`
      ready + protocol documented; run `sudo bash scripts/lock_clocks.sh lock`
      (needs root) before the Phase 1 microbenchmarks
- [x] First commit of RESULTS.md with the vLLM + PyTorch baselines

**Exit criterion:** baselines reproducible to <2% run-to-run; harness trusted.

---

## Phase 1 — Fused attention  (design doc: `docs/01-fused-attention.md`)

Current state on `main`: **v3 (single-warp block + vectorized 64-bit KV
loads), 0.713 ms / 189 GB/s** at the reference workload — 2.34× over v0,
1.91× faster than PyTorch SDPA. See
[`docs/01-fused-attention-journey.md`](docs/01-fused-attention-journey.md)
for the full v0–v5 narrative (theory, design, result, lesson per step,
including the v4/v5 explorations that didn't land).

### 1a. Reference + correctness
- [x] `reference/attention_ref.py`: naive eager attention (QK^T, softmax, ·V)
- [x] Reference wrapper around `F.scaled_dot_product_attention`
- [x] `tests/test_attention.py`: shapes/dtypes/tolerances defined  (rtol/atol = 2e-2, configs `batch ∈ {1, 8} × seqlen_kv ∈ {128, 2048}`; green on every landed kernel)

### 1b. Naive CUDA kernel  (CUDA vs CUDA — baseline)
- [x] Decode kernel: one block per (batch, head); two-pass softmax; correct  (v0, commit `46930c2`)
- [x] Wire into the PyTorch extension (`bindings/`, `setup.py`)  (also fixed pre-existing relative-include-path bug in setup.py)
- [x] `test_attention.py` green against the reference  (max |abs diff| 6.1e-5)
- [x] RESULTS.md: naive kernel latency + achieved bandwidth  (1.669 ms, 80 GB/s)

### 1c. Optimization sweep  (CUDA vs CUDA — one commit per step)
- [x] Online (streaming) softmax — single pass  (v1: `46ae1ea` naive port regressed; `ad9c57f` V-prefetch fix → 1.637 ms / 82 GB/s)
- [x] Warp-level reductions for dot-products and softmax stats  (v2, commit `db6ab0b` — single-sync block reduce via double-buffered shmem → 1.069 ms / 126 GB/s)
- [x] Vectorized 128-bit (`float4`) coalesced KV loads  (v3, commit `ccdb6df` — landed as 64-bit per-thread `uint2` for clean head_dim=128 mapping with a single-warp block → 0.713 ms / 189 GB/s. Going to true 128-bit per-thread vec would require multi-`j`-per-iter state management, deferred.)
- [~] Split-K over the KV sequence + partial-result combine (FlashDecoding)  (v4 explored at `f904aae`, reverted in `e39c97f` — regressed across every batch size tested; our workload was per-SM bandwidth-bound, not grid-undersized)
- [~] `cp.async` double-buffering of KV tiles  (v5 explored at `78a28ff`, reverted in `4254ce2` — regressed by ~50 µs; nvcc was already pipelining v3's loads implicitly, so the explicit pipeline cost more in shmem hop than it saved in overlap)
- [ ] (stretch) Tensor Core MMA path; (stretch) prefill FA-2 forward kernel
- [ ] After each step: `ncu` profile, log before/after in RESULTS.md  (every "Cause" cell in RESULTS.md still says "ncu pending"; needs the GPU clocks locked first)

### 1d. Compare + write up  (CUDA vs Python / SOTA)
- [~] Benchmark vs PyTorch eager, `F.sdpa`, `flash_attn`  (eager + SDPA done in `benchmarks/bench_attention.py`; `flash_attn` direct row in RESULTS.md still TBD — SDPA dispatches to FA/cuDNN so we have an indirect read)
- [ ] Roofline placement; explain the gap to SOTA with profiler metrics  (we're at 189/1008 GB/s ≈ 19% of peak HBM; no roofline plot yet, no `ncu` metrics)
- [x] Findings section in `docs/01-fused-attention.md`  (summary + 6 lessons; full narrative in `docs/01-fused-attention-journey.md`)

**Exit criterion:** Track 1 *Target* met (within 20% of `flash_attn` on
achieved bandwidth — see `docs/00`), write-up complete.  We beat
PyTorch SDPA by 1.91× and are 1.27× over the *Threshold* (≥3× PyTorch
eager — we're at 5.3×); the *Target* comparison vs raw `flash_attn`
remains pending.

---

## Phase 2 — KV-cache compression  (design doc: `docs/02-kv-cache-compression.md`)

- [ ] Reference: INT8/INT4 quantize + dequantize in PyTorch; perplexity harness
- [ ] CUDA quantize kernel (called on KV-cache append)
- [ ] Fuse dequant into the Phase 1 attention kernel (read INT8 KV directly)
- [ ] `tests/test_kv_cache.py`: correctness + accuracy (perplexity delta)
- [ ] INT4 path: per-channel K scales, per-token V scales (KIVI-style)
- [ ] Measure: memory reduction, perplexity delta, decode tokens/sec
- [ ] RESULTS.md entries; findings in `docs/02`

**Exit criterion:** Track 2 *Threshold* met (Target if time allows).

---

## Phase 3 — Quantized matmul  (design doc: `docs/03-quantized-matmul.md`)

- [ ] Reference: W4A16 quantize + dequant-and-matmul in PyTorch
- [ ] Naive CUDA W4A16 GEMM: unpack INT4, dequant in registers, accumulate
- [ ] `tests/test_quant.py`: correctness vs reference
- [ ] Optimize: group-wise scales, shared-memory staging, vectorized unpack
- [ ] Tensor Core path for compute-bound (prefill) shapes
- [ ] Benchmark vs FP16 cuBLAS and vs Marlin on decode + prefill shapes
- [ ] RESULTS.md entries; findings in `docs/03`

**Exit criterion:** Track 3 *Threshold* met (Target if time allows).

---

## Phase 4 — End-to-end integration & presentation

- [ ] Monkeypatch Llama 3 8B (HF) to call the custom attention + GEMM kernels
- [ ] Enable KV-cache compression in the patched model
- [ ] End-to-end tokens/sec + peak memory vs vanilla vLLM, same workload
- [ ] Quality eval: perplexity / chat eval with compression on
- [ ] Generate roofline plots + `ncu`/`nsys` summary figures
- [ ] Fill the README results table
- [ ] Distill RESULTS.md into a slide-ready interview summary

**Exit criterion:** End-to-end *Target* met or a profiler-backed account of why.

---

## Stretch — AMD / HIP portability study

- [ ] `hipify` the optimized CUDA kernels; resolve portability gaps
- [ ] Benchmark on an AMD GPU (ROCm); compare achieved bandwidth vs NVIDIA
- [ ] Write up the "C++/HIP vs CUDA" cross-vendor comparison
