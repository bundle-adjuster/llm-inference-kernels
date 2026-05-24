# Results Log

The incremental record of every baseline and every optimization step. This is
the primary interview artifact — keep it honest, current, and profiler-backed.

One row per optimization step. "Cause" must cite an Nsight Compute metric, not a
guess. Commit hash makes every number reproducible.

## Environment

See [`env-report.md`](env-report.md). RTX 4090 (sm_89), CUDA 12.4, torch
2.5.1+cu124, transformers 4.47.1, vLLM 0.6.6, flash-attn 2.8.3. Clocks: default
(boost) for the Phase 0 end-to-end baselines below — observed run-to-run
variance < 0.2%; lock via `scripts/lock_clocks.sh` before the Phase 1
microbenchmarks. Reference workload defined in
[`../benchmarking-methodology.md`](../benchmarking-methodology.md).

## Baselines (Phase 0)

Reference serving workload: Llama 3.1 8B Instruct FP16, batch 16, prompt 512,
generate 512 — 8192 output tokens/run, greedy, EOS suppressed. Measured by
`benchmarks/bench_e2e.py`: median of 3 timed runs after 1 warmup.

| What | Latency (median) | Throughput | Peak VRAM | Notes |
|------|------------------|------------|-----------|-------|
| PyTorch (HF `generate`) | 23.10 s | 354.6 tok/s | 18.50 GB | sdpa attention; static batch, no continuous batching |
| vanilla vLLM 0.6.6 | 11.65 s | 703.2 tok/s | n/a | the bar; paged KV + continuous batching + CUDA graphs; `max_model_len=1024` |

vLLM delivers **1.98×** the HF `generate()` throughput on this workload. vLLM
peak VRAM is not directly comparable — it reserves a fixed `gpu_memory_utilization`
(0.9 × 24 GB) pool up front by design.

## Track 1 — Fused attention

Microbench workload: Llama 3 8B head config (`n_heads=32, n_kv_heads=8,
head_dim=128`), `batch=8, seqlen_kv=4096`, fp16. Measured by
`benchmarks/bench_attention.py` (CUDA events, 25 warmup + 100 timed).
Achieved BW = `(|K|+|V|) / median_latency`; each tensor is 64 MB so 128 MB
streamed per call. For reference on this workload: PyTorch eager 3.77 ms,
PyTorch SDPA 1.36 ms. Clocks not yet locked (TODO before `ncu` runs).

| Step | Commit | Latency | Achieved BW | Speedup vs prev | Cause (ncu metric) |
|------|--------|---------|-------------|-----------------|--------------------|
| v0 naive (two-pass softmax) | 46930c2 | 1.669 ms | 80 GB/s | — (baseline) | per-`j` block reductions in phase 1 serialize compute; KV is read in two passes (K in phase 1, V in phase 3); GQA reads not de-duplicated across query heads sharing a kv head. ncu profile pending (lock clocks first). |
| + online softmax (v1) | _HEAD_ | 2.078 ms | 65 GB/s | **0.80× (regression)** | textbook Milakov–Gimelshein recurrence per-thread. Two regressions vs v0: (1) **redundant exp work** — every thread re-computes the same scalar `alpha = exp(m_old−m_new)` and `p_j = exp(s_j−m_new)`, so the block does ~128× as many `__expf` calls as v0's phase 2; (2) **lost streaming pipeline** — v0's phase 3 is a sync-free `o += s[j]·v[j,tid]` loop the compiler can unroll/pipeline, whereas v1 interleaves V loads with the per-`j` sync, blocking V prefetch. The single-pass + unbounded-seqlen properties are still wins; v2 fixes the redundancy. ncu pending. |
| + warp reductions | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + vectorized loads | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + split-K (FlashDecoding) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + cp.async double-buffer | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

Correctness: v0 max |abs diff| vs fp32 reference = 6.1e-5 (well under the
2e-2 gate). `tests/test_attention.py` green at batch ∈ {1, 8} ×
seqlen_kv ∈ {128, 2048}.

**vs SOTA:** `flash_attn` decode = _TBD_ µs. Gap = _TBD_%. Explanation: _TBD_.

## Track 2 — KV-cache compression

| Variant | Commit | Memory | Perplexity Δ | Decode tok/s | Notes |
|---------|--------|--------|--------------|--------------|-------|
| FP16 KV (baseline) | _TBD_ | 1.0× | 0.0 | _TBD_ | |
| INT8 KV | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| INT4 KV (per-channel K) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |

## Track 3 — Quantized matmul (W4A16)

| Shape (M,K,N) | Commit | FP16 cuBLAS | This kernel | Speedup | % of Marlin |
|---------------|--------|-------------|-------------|---------|-------------|
| _TBD decode_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| _TBD prefill_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## End-to-end (Phase 4)

| Config | Commit | Tokens/sec | Peak memory | vs vLLM | Quality |
|--------|--------|-----------|-------------|---------|---------|
| vanilla vLLM | _TBD_ | _TBD_ | _TBD_ | 1.00× | ref |
| custom kernels | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + KV compression | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
