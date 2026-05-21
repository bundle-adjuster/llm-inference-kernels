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

| Step | Commit | Latency | Achieved BW | Speedup vs prev | Cause (ncu metric) |
|------|--------|---------|-------------|-----------------|--------------------|
| Naive CUDA decode kernel | _TBD_ | _TBD_ | _TBD_ | — (baseline) | — |
| + online softmax | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + warp reductions | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + vectorized loads | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + split-K (FlashDecoding) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| + cp.async double-buffer | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

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
