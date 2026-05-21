# TODO — phased, step-by-step plan

Work top to bottom. Each phase has an **exit criterion** — do not start the next
phase until it is met. Check items off as you land them; each optimization step
is its own commit + a row in `docs/results/RESULTS.md`.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Environment, baselines, infrastructure

- [x] Create the reproducible conda env: `conda env create -f environment.yml`
      (Python 3.11, CUDA 12.4 toolkit; target GPU RTX 4090 / sm_89)
- [x] Install flash-attn: `MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation`
- [x] Confirm `torch.cuda.is_available()` and the cu124 build inside the env
- [x] After install succeeds, lock it:
      `conda env export --no-builds > environment.lock.yml`
- [x] Run `scripts/detect_env.sh` → `docs/results/env-report.md`
- [ ] Download Llama 3 8B weights; confirm it loads in FP16 and generates text
- [ ] Stand up vLLM; record baseline tokens/sec on the reference workload
- [~] Verify `benchmarks/harness.py` — timing path verified; allclose/memory with first kernel
- [x] Lock the reference workload (batch, prompt len, gen len) in
      `docs/benchmarking-methodology.md`
- [ ] Lock GPU clocks; document the measurement protocol
- [ ] First commit of RESULTS.md with the vLLM + PyTorch baselines

**Exit criterion:** baselines reproducible to <2% run-to-run; harness trusted.

---

## Phase 1 — Fused attention  (design doc: `docs/01-fused-attention.md`)

### 1a. Reference + correctness
- [ ] `reference/attention_ref.py`: naive eager attention (QK^T, softmax, ·V)
- [ ] Reference wrapper around `F.scaled_dot_product_attention`
- [ ] `tests/test_attention.py`: shapes/dtypes/tolerances defined

### 1b. Naive CUDA kernel  (CUDA vs CUDA — baseline)
- [ ] Decode kernel: one block per (batch, head); two-pass softmax; correct
- [ ] Wire into the PyTorch extension (`bindings/`, `setup.py`)
- [ ] `test_attention.py` green against the reference
- [ ] RESULTS.md: naive kernel latency + achieved bandwidth

### 1c. Optimization sweep  (CUDA vs CUDA — one commit per step)
- [ ] Online (streaming) softmax — single pass
- [ ] Warp-level reductions for dot-products and softmax stats
- [ ] Vectorized 128-bit (`float4`) coalesced KV loads
- [ ] Split-K over the KV sequence + partial-result combine (FlashDecoding)
- [ ] `cp.async` double-buffering of KV tiles
- [ ] (stretch) Tensor Core MMA path; (stretch) prefill FA-2 forward kernel
- [ ] After each step: `ncu` profile, log before/after in RESULTS.md

### 1d. Compare + write up  (CUDA vs Python / SOTA)
- [ ] Benchmark vs PyTorch eager, `F.sdpa`, `flash_attn`
- [ ] Roofline placement; explain the gap to SOTA with profiler metrics
- [ ] Findings section in `docs/01-fused-attention.md`

**Exit criterion:** Track 1 *Target* met (see `docs/00`), write-up complete.

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
