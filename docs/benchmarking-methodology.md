# Benchmarking Methodology

Fix this protocol *before* recording any number. An unfair benchmark is worse
than no benchmark — in an interview it is a liability.

## Golden rule

**Correctness gates performance.** No latency/throughput number is recorded for
a kernel until it passes `tests/` against the PyTorch reference within the
documented tolerance. A fast wrong kernel scores zero.

## Timing

- Use **CUDA events** (`torch.cuda.Event` / `cudaEventRecord`), never wall clock.
- **Warm up** ≥ 25 iterations before timing (JIT, caches, clocks).
- Time ≥ 100 iterations; report **median** + p10/p90, not just mean.
- `torch.cuda.synchronize()` around the timed region.
- Report run-to-run variance; investigate anything above ~3%.

## GPU state (reproducibility)

- Lock clocks: `sudo nvidia-smi -lgc <freq>` and `-lmc <freq>`; record the
  values in `env-report.md`. Unlocked clocks make numbers irreproducible.
- Record GPU temperature; thermal throttling silently corrupts long runs.
- One benchmark process at a time; nothing else on the GPU.
- Pin the driver / CUDA / PyTorch / vLLM / flash-attn versions in the report.

## Fair comparison

- **Same precision** across compared implementations, or the difference is
  stated loudly (e.g. W4A16 vs FP16 — the whole point, so label it).
- **Same problem sizes** — identical shapes, sequence lengths, batch.
- **Same workload** — see the reference workload below.
- Exclude one-time setup (allocation, weight load) from the timed region;
  include everything that recurs per step.
- For end-to-end vs vLLM: same model, same sampling params, same input/output
  lengths, same batch. Document anything that cannot be matched.

## Metrics to report

| Metric | Where it matters |
|--------|------------------|
| Latency (median µs/ms) | every kernel |
| Throughput (tokens/sec) | end-to-end |
| Achieved HBM bandwidth (% of peak) | memory-bound kernels (attention, decode GEMM) |
| Achieved TFLOP/s (% of peak) | compute-bound kernels (prefill) |
| Peak memory (`torch.cuda.max_memory_allocated`) | KV-cache / quantization tracks |
| Accuracy (perplexity delta) | compression / quantization tracks |
| Occupancy, warp-stall reasons | every optimization step (from `ncu`) |

## Roofline

Place every kernel on a roofline plot (arithmetic intensity vs achieved
FLOP/s). It shows whether a kernel is memory- or compute-bound and how much
headroom remains — and turns "it got faster" into "it moved from X to Y on the
roofline because Z."

## Profiling

- **Nsight Compute (`ncu`)** — per-kernel: occupancy, memory throughput,
  achieved vs peak, stall reasons. Run after *every* optimization step.
- **Nsight Systems (`nsys`)** — end-to-end timeline: kernel gaps, launch
  overhead, H2D/D2H copies, CPU/GPU overlap.
- Save raw reports under `docs/results/raw/` (git-ignored); commit the
  distilled numbers to `RESULTS.md`.

## Reference workload (LOCKED 2026-05-21)

Defined once and reused everywhere so all numbers are comparable. Frozen for the
project — changing any value invalidates previously recorded numbers.

- **Model:** Llama 3.1 8B Instruct, FP16 base.
- **End-to-end serving workload:**
  - Batch size: **16**
  - Prompt length: **512** tokens
  - Generation length: **512** tokens
  - Footprint: ~16 GB weights + ~4 GB KV cache — fits the 24 GB RTX 4090.
- **Decode-attention microbenchmark:** batch **8**, KV-cache length sweep
  ∈ **{512, 1024, 2048, 4096, 8192, 16384}**.
- **Quantized-matmul microbenchmark:** real Llama 3.1 8B linear-layer shapes,
  M ∈ **{1, 8, 32, 128, 512}** (crosses the memory- → compute-bound boundary).

## The incremental log

Every optimization step appends a row to `docs/results/RESULTS.md`:
*step name · commit hash · before → after · speedup · the `ncu` metric that
explains it.* That log is the primary interview artifact — keep it honest and
current.
