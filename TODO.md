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

Current state on `main`: **v6 (FlashDecoding split-K on multi-warp blocks —
4 warps/block, 4-deep unrolled KV loads), 155.6 µs at the reference workload
(batch=8, kv_len=4096) — 1.01× vs GQA-native PyTorch SDPA (it *beats* SDPA),
4.59× over the old v3, ~82% of peak HBM.** The binding `decode_attention`
now dispatches to v6; the old kernel is preserved as `decode_attention_v3`.
The old v3 (single-warp block, 0.713 ms / 189 GB/s, 2.34× over v0) was
**retired in Phase 7**: measured against a *fair* GQA-native baseline
(`F.scaled_dot_product_attention(..., enable_gqa=True)`), v3 was 4.55×
*slower* (0.22×) because its single-warp block is **occupancy-bound**
(fills ~2 of 128 SMs), not bandwidth-bound. Phase 8's v6 fixes exactly that.
See [`docs/06-attention-splitk-journey.md`](docs/06-attention-splitk-journey.md)
for the fix and
[`docs/05-baseline-correction-journey.md`](docs/05-baseline-correction-journey.md)
for the baseline correction; the full v0–v5 narrative is in
[`docs/01-fused-attention-journey.md`](docs/01-fused-attention-journey.md)
(theory, design, result, lesson per step).

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
- [x] Split-K over the KV sequence + partial-result combine (FlashDecoding)  (v4 explored at `f904aae`, reverted in `e39c97f`; the "per-SM bandwidth-bound, not grid-undersized" diagnosis was **wrong** — the real limiter was occupancy (the single-warp block filled ~2 of 128 SMs). Phase 8's **v6** re-did split-K on multi-warp blocks (4 warps/block) in `kernels/attention/fused_attention_splitk.cu` and landed: 155.6 µs, 4.59× over v3, beats fair GQA-native SDPA — see `docs/06-attention-splitk-journey.md`)
- [~] `cp.async` double-buffering of KV tiles  (v5 explored at `78a28ff`, reverted in `4254ce2` — regressed by ~50 µs; nvcc was already pipelining v3's loads implicitly, so the explicit pipeline cost more in shmem hop than it saved in overlap)
- [ ] (stretch) Tensor Core MMA path; (stretch) prefill FA-2 forward kernel
- [ ] After each step: `ncu` profile, log before/after in RESULTS.md  (every "Cause" cell in RESULTS.md still says "ncu pending"; needs the GPU clocks locked first)

### 1d. Compare + write up  (CUDA vs Python / SOTA)
- [x] Benchmark vs PyTorch eager, `F.sdpa`, `flash_attn`  (Phase 8: v6 measured against **GQA-native** `F.scaled_dot_product_attention(..., enable_gqa=True)`, which dispatches to FlashAttention/cuDNN — v6 = 1.01× on the reference workload (batch=8, kv_len=4096), i.e. within 20% of flash. See `docs/06-attention-splitk-journey.md`. The earlier eager/SDPA rows fed SDPA a 4×-expanded GQA KV cache and were **retired in Phase 7** — see `docs/05-baseline-correction-journey.md`.)
- [ ] Roofline placement; explain the gap to SOTA with profiler metrics  (Phase 8 v6 reaches ~82% of peak HBM on the HBM-bound reference workload — up from v3's 189/1008 GB/s ≈ 19%; the honest remaining gap is L2-resident shapes (kv ≤ 1024), where v6 trails flash's L2 blocking at 0.69–0.82×. No roofline plot yet, no `ncu` metrics — see `docs/06-attention-splitk-journey.md`)
- [x] Findings section in `docs/01-fused-attention.md`  (summary + 6 lessons; full narrative in `docs/01-fused-attention-journey.md`)

**Exit criterion:** Track 1 *Target* met — **Phase 8's v6 split-K beats a
fair, GQA-native SDPA baseline (1.01× on the reference workload, i.e. within
20% of `flash_attn`; 4.59× over the old v3)**, and still clears the
*Threshold* (≥3× PyTorch eager). The old "we beat PyTorch SDPA by 1.91×"
claim was **retired in Phase 7**: that baseline fed SDPA a 4×-expanded GQA
KV cache; against GQA-native SDPA the v3 kernel was 4.55× *slower*
(occupancy-bound, not bandwidth-bound). v6 is what actually met the Target —
see [`docs/06-attention-splitk-journey.md`](docs/06-attention-splitk-journey.md).
Remaining honest gap: L2-resident shapes (kv ≤ 1024), where v6 trails at
0.69–0.82×.

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

## Phase 4 — End-to-end integration  *(complete; full narrative in `docs/04-end-to-end-integration-journey.md`)*

All four sub-phases merged to `main`. Headline numbers in
[`docs/results/RESULTS.md`](docs/results/RESULTS.md) "End-to-end (Phase 4)"
section; per-config JSON outputs in `docs/results/{lm_eval,e2e_eval}/`.

- [x] **4-prep** (`phase4-eval-prep`, commit `1c56d82`): eval scaffolding
  (`scripts/run_lm_eval.py` + `scripts/run_e2e_eval.py`) + vanilla baseline
  numbers locked. Llama 3.1 8B Instruct: MMLU 68.32%, HellaSwag 79.51%,
  ARC-C 60.84%, PPL 7.055, 335.8 tok/s, 18.50 GB peak VRAM.
- [x] **4a** (`phase4-attention`, commit `d6d8c88`): F.sdpa rebind →
  Phase 1 v3 decode_attention for q_len==1; un-expands `repeat_kv`. **Bit-
  identical accuracy** + 1.025× tok/s.
- [x] **4b** (`phase4-kv-int4`, commit `0836a77`): Int4KIVICache + INT4
  decode forward replacement + KIVI F.sdpa rebind for lm-eval. **1.55×
  tok/s** for −1.03 pp MMLU + Δppl +0.20 (matches Phase 2c kernel-level).
- [x] **4c** (`phase4-w4a16`, commit `8882880`): W4A16 patched Linears
  (224 in-place replacements). **−51% peak VRAM** (18.50 → 9.05 GB) +
  1.16× tok/s at batch=1; regression at batch=16 because the Phase 3
  kernel is M=1-only → Phase 5.
- [x] **4d** (`phase4-wrapup`): journey doc, README headline numbers,
  TODO update.

**What's still open** (deferred, see journey doc "What's still open"):
- `ncu` profile + roofline plot for the patched decode step
- Calibration-aware quant (GPTQ / AWQ) to recover the W4A16 accuracy hit
- vLLM port of the patches

---

## Phase 5 — Batched-decode W4A16 kernel  *(partial: recovery yes, full win no)*

Resolves the Phase 4c regression at batch=16. Same K-split-across-warps
pattern as Phase 3c, but each thread accumulates a `BLOCK_M=16`-length
vector of fp32 partials; each warp has its own `[BLOCK_M, group_size]`
activation tile in shmem so the int4 weight stream is amortized across
all M rows.

- [x] `w4a16_gemm_batched_decode_kernel` in `kernels/quant/quant_matmul.cu`
- [x] Launcher dispatch: M==1 → Phase 3c; M∈[2,16] → Phase 5; M>16 → Phase 3b naive
- [x] Correctness vs reference at M ∈ {1, 2, 4, 8, 16}: max abs err 0.25,
  mean rel err ~0.001 (same fp16 reduction noise floor as Phase 3c)
- [x] Benchmark vs cuBLAS at M ∈ {1, 4, 8, 16, 32} on all three shapes:
  M=2..16 is near-flat ~200 µs (amortization works structurally), but
  remains ~1.7× behind cuBLAS at M=16 because cuBLAS uses tensor cores
- [x] Re-run Phase 4c at locked batch=16: **40.9 → 199.9 tok/s (4.9×
  recovery; 0.60× vs vanilla HF)**. Memory savings (51% peak VRAM) and
  all accuracy numbers unchanged.
- [x] Update `docs/04-end-to-end-integration-journey.md` (Phase 5
  addendum), RESULTS.md Phase 4c row + Phase 5 update, README headline.

**Exit criterion (partial):** the regression is largely recovered (0.12×
→ 0.60× vs vanilla); a full win at batch=16 requires tensor cores —
Phase 6.

---

## Phase 6 — Tensor-core W4A16 (`mma.sync`)  *(complete; kernel win, no e2e move)*

Built `w4a16_gemm_tc_kernel` using the wmma C++ API
(`mma.sync.m16n16k16` fp16->fp32). Launcher routes M ∈ [2, 16] to v3.

- [x] WMMA fragments + `load_matrix_sync` / `mma_sync` inner loop
- [x] Correctness vs reference at M ∈ {1, 2, 4, 8, 16}: max abs 0.25,
  mean rel ~0.0001 (better than v2)
- [x] Benchmark: 1.3-1.4× over Phase 5 v2 across all Llama 3 shapes
  (e.g. QKV/O M=16: 211 → 152 µs)
- [x] Re-run Phase 4c: 199.9 → 198.7 tok/s — *unchanged* despite the
  kernel win. Host-side Python/dispatch overhead now occupies the gap.

**Outcome:** kernel quality improved; e2e at the locked workload is
bottlenecked elsewhere. Closing the e2e gap is now a host-side problem.

---

## Phase 7 — Hide host overhead at decode

Phase 6's kernel win is being absorbed by per-`nn.Module.__call__`
Python dispatch overhead (estimated 5–30 µs × 224 Linears × 512 decode
steps ≈ 5–30 s of host work across the run, on a total run of ~41 s).
Several mutually compatible approaches:

- [ ] **CUDA graphs.** Capture one decode step with
  `torch.cuda.graph(...)` and replay it. The graph replays as a single
  GPU command, hiding all the Python overhead between kernel launches.
  Per-decode-step CUDA graph capture is what production decode engines
  (vLLM, TGI) use.
- [ ] **`torch.compile(model, mode="reduce-overhead")`.** Same idea
  via TorchDynamo + CUDA graphs. Less invasive than manual capture; some
  graph-break friction with our monkey-patched `LlamaSdpaAttention.forward`.
- [ ] **`ncu` profile** of one decode step to actually confirm whether
  the bottleneck is host overhead (kernel gaps in the timeline) vs
  device-side stalls.

**Exit criterion:** Phase 4c at batch=16 closes the gap to vanilla
(>= 300 tok/s) while keeping the 51% peak VRAM win and the Phase 6
accuracy numbers.

---

## Phase 8 — Further W4A16 kernel work *(only if Phase 7 unblocks visible wins)*

The remaining ~1.7× kernel-level gap to cuBLAS (QKV/O and MLP down)
would need Marlin-style optimizations:

- [ ] Dequantize directly into MMA-register layout (skip the shmem hop +
  one `__syncthreads`)
- [ ] Double-buffer the weight shmem so dequant for group g+1 overlaps
  the MMA for group g
- [ ] Maybe larger BLOCK_M (32) with BLOCK_K tiling

---

## Stretch — AMD / HIP portability study

- [ ] `hipify` the optimized CUDA kernels; resolve portability gaps
- [ ] Benchmark on an AMD GPU (ROCm); compare achieved bandwidth vs NVIDIA
- [ ] Write up the "C++/HIP vs CUDA" cross-vendor comparison
