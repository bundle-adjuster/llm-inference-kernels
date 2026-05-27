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

Current state on `main`: **INT8 KV essentially lossless (Δppl +0.0008,
0.51× memory, tied latency); INT4 KIVI clears the < 0.5 target (Δppl
+0.196, 0.27× memory, 1.29× faster than v3).** Both threshold AND target
hit. Full Findings in [`docs/02-kv-cache-compression.md`](docs/02-kv-cache-compression.md).

- [x] Reference: INT8/INT4 quantize + dequantize in PyTorch (`reference/kv_cache_ref.py`); perplexity harness (`scripts/eval_perplexity.py`)
- [x] CUDA quantize kernels: INT8 per-token, INT4 K per-channel groupwise, INT4 V per-token, all packed (`kernels/kv_cache/kv_compress.cu`)
- [x] Fuse dequant into the Phase 1 attention kernel: INT8 reads (`kernels/attention/fused_attention_int8.cu`), INT4 KIVI reads (`kernels/attention/fused_attention_int4.cu`); both built on v3, both pass correctness
- [x] `tests/test_kv_cache.py`: 38 tests covering reference round-trip + CUDA-vs-reference for all 5 paths
- [x] INT4 path: per-channel K scales (group_size=32), per-token V scales (KIVI-style)
- [x] Measure: memory reduction (`benchmarks/bench_kv_cache.py`), perplexity delta (`scripts/eval_perplexity.py`); decode tokens/sec deferred to Phase 4 (requires plumbing INT4 attention into Llama's actual decode loop)
- [x] RESULTS.md entries; findings in `docs/02`

**Exit criterion:** Track 2 *Threshold* AND *Target* met (Δppl < 0.2 for
INT8 ✓ 0.0008; Δppl < 0.5 for INT4 KIVI ✓ 0.196).

---

## Phase 3 — Quantized matmul  (design doc: `docs/03-quantized-matmul.md`)

Current state on `main`: **Phase 3 *Target* hit. All three M=1 Llama 3 8B
layer shapes beat fp16 cuBLAS by 2.88–6.97×.** Full narrative in
[`docs/03-quantized-matmul-journey.md`](docs/03-quantized-matmul-journey.md).

- [x] Reference: W4A16 quantize + dequant-and-matmul in PyTorch (`reference/quant_matmul_ref.py`, symmetric per-channel groupwise INT4, group_size=128 along K)
- [x] Naive CUDA W4A16 GEMM: unpack INT4, dequant in registers, accumulate (3b, kernel `w4a16_gemm_naive_kernel`)
- [x] `tests/test_quant.py`: 24 tests covering reference round-trip + matmul-vs-fp16 noise bounds + CUDA-vs-reference equivalence + pack/unpack roundtrip
- [x] Optimize: K-split across 4 warps + activations cached in shmem (3c, kernel `w4a16_gemm_decode_kernel`; launcher dispatches M==1 → decode, else → naive)
- [ ] Tensor Core path for compute-bound (prefill) shapes  *(stretch, deferred)*
- [~] Benchmark vs FP16 cuBLAS on decode shapes (`benchmarks/bench_w4a16.py`); vs Marlin deferred
- [x] RESULTS.md entries; findings in `docs/03`

**Exit criterion:** Track 3 *Target* met (2–3× over fp16 cuBLAS on
decode shapes — landed at 2.88×–6.97×).

---

## Phase 4 — End-to-end integration & presentation

Approach: monkeypatch HF Llama 3.1 8B Instruct one kernel at a time, on its
own branch, with full eval per step (the GPTQ/AWQ/KIVI paper bar — MMLU 5-shot,
HellaSwag 0-shot, ARC-Challenge 25-shot, plus WikiText-2 PPL, greedy-token
match rate, decode tokens/sec, peak VRAM). Each integration step gets its own
RESULTS.md row so the accuracy/latency/memory tradeoff is attributable.

### 4-prep. Eval infrastructure (branch `phase4-eval-prep`)
- [ ] Install `lm-evaluation-harness`; smoke-test on Llama 3.1 8B Instruct
- [ ] `scripts/run_lm_eval.py`: wrapper for MMLU/HellaSwag/ARC-C
- [ ] `scripts/run_e2e_eval.py`: WikiText-2 PPL + greedy-token-match + tokens/sec + peak VRAM
- [ ] Run vanilla Llama 3.1 8B Instruct baseline; lock numbers in `docs/results/lm_eval/vanilla.json` and `docs/results/e2e_eval/vanilla.json`; first Phase 4 row in RESULTS.md
- [ ] Save vanilla greedy-reference token outputs (used by 4a–4c match rate)
- [ ] Merge `phase4-eval-prep` → `main`

### 4a. Attention kernel integration (branch `phase4-attention`)
- [ ] Monkeypatch `F.scaled_dot_product_attention` (same hook as `eval_perplexity.py`): dispatch our Phase 1 v3 `decode_attention` for `q_len == 1` (decode), fall back to the original SDPA for prefill
- [ ] Greedy-match on 10 fixed prompts vs vanilla reference (catch integration bugs)
- [ ] Full eval suite — lm-eval + e2e
- [ ] RESULTS.md row + journey doc section
- [ ] Merge → `main`

### 4b. KV-cache compression integration (branch `phase4-kv-int4`)
- [ ] Replace HF `DynamicCache` with an INT4 KIVI cache: per-channel groupwise K (group=32), per-token V, packed 4-bit
- [ ] On append: quantize current-token K/V via our Phase 2 kernels
- [ ] On read: dispatch our Phase 2 INT4 KIVI attention kernel for decode
- [ ] Full eval suite — Δppl, Δ MMLU/HellaSwag/ARC-C, tokens/sec, peak VRAM
- [ ] RESULTS.md row + journey doc section
- [ ] Merge → `main`

### 4c. W4A16 weight integration (branch `phase4-w4a16`)
- [ ] Offline weight quantization script (`scripts/quantize_llama_weights.py`): symmetric INT4, group=128 along K, save packed + scales for each `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`
- [ ] Patched linear layers using our Phase 3 `w4a16_gemm` for decode (M=1); fp16 fallback for prefill
- [ ] Full eval suite — Δ accuracy on all metrics, tokens/sec, peak VRAM (~10 GB savings expected)
- [ ] RESULTS.md row + journey doc section
- [ ] Merge → `main`

### 4d. Final headline + presentation (branch `phase4-wrapup` or directly on `main`)
- [ ] `docs/04-end-to-end-integration-journey.md` — full Phase 4 narrative
- [ ] Roofline plot + `ncu`/`nsys` summary figures
- [ ] Fill README results table with final headline numbers
- [ ] Distill RESULTS.md into a slide-ready interview summary

**Exit criterion:** End-to-end *Target* met or a profiler-backed account of
why; accuracy tradeoff documented across all four eval metrics for each
kernel.

---

## Stretch — AMD / HIP portability study

- [ ] `hipify` the optimized CUDA kernels; resolve portability gaps
- [ ] Benchmark on an AMD GPU (ROCm); compare achieved bandwidth vs NVIDIA
- [ ] Write up the "C++/HIP vs CUDA" cross-vendor comparison
