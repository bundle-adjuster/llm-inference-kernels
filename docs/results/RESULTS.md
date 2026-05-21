# Results Log

The incremental record of every baseline and every optimization step. This is
the primary interview artifact — keep it honest, current, and profiler-backed.

One row per optimization step. "Cause" must cite an Nsight Compute metric, not a
guess. Commit hash makes every number reproducible.

## Environment

See [`env-report.md`](env-report.md). Clocks locked: _TBD_. Reference workload:
_TBD_ (defined in `../benchmarking-methodology.md`).

## Baselines (Phase 0)

| What | Workload | Number | Notes |
|------|----------|--------|-------|
| PyTorch eager (Llama 3 8B) | reference workload | _TBD_ tok/s | |
| vanilla vLLM (Llama 3 8B) | reference workload | _TBD_ tok/s | the bar |

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
