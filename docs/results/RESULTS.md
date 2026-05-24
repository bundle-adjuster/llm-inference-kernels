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
| + online softmax (naive port) | 46ae1ea | 2.078 ms | 65 GB/s | **0.80× (regression)** | Textbook Milakov–Gimelshein recurrence per-thread. The V load is issued at point of use, *inside* the per-`j` `__syncthreads()` barriers — so its latency can't hide behind the K reduction. v0 doesn't suffer this because its phase 3 V-loop is sync-free. (Also suspected at the time: ~128× redundant `__expf` calls, since the scalar `alpha`/`p_j` recurrence is duplicated across threads — but moving it to lane 0 + shmem broadcast made things *worse* (2.26 ms), confirming SIMT-redundant ALU work is essentially free.) |
| + V-load prefetch | ad9c57f | 1.637 ms | 82 GB/s | 1.27× over naive port; 1.02× over v0 | Issue `v_j = V[j, tid]` at the *top* of the iteration, alongside `k_j`. The load is non-blocking; the value isn't consumed until after both `__syncthreads()` and the softmax update, so the V latency hides behind ~all of that work. nvcc would not hoist the load above the syncs on its own. Now consistent with theory: single-pass + no shmem score buffer + sync-overlapped V latency. ncu pending. |
| v2 (single-sync block reduce) | db6ab0b | 1.069 ms | 126 GB/s | **1.53× over v1-prefetch; 1.56× over v0** | Two changes drop the per-`j` sync count from 2 to 1. (a) All warps redundantly do the final cross-warp reduce — read `reduce_smem[buf][0..n_warps]`, mask, `warp_reduce_sum`. Every thread arrives at the same `s_j` via warp shuffle, no shmem broadcast. (b) Double-buffer `reduce_smem` on `j & 1` so iter j+1's write never races iter j's still-in-flight reads. The hazard between iter j's read and iter j+2's write (same slot) is gated by iter j+1's sync. The speedup is larger than the saved sync alone would predict (~150µs predicted vs 570µs measured) — removing the `s_bcast` shmem hop also lets `s_j` flow directly from the shfl tree into the softmax FMAs. Now beating PyTorch SDPA (1.36 ms) by 1.27×. ncu pending. |
| v3 (vectorized KV loads) | ccdb6df | 0.713 ms | 189 GB/s | **1.50× over v2; 2.34× over v0** | Two changes: (a) block shrinks 128→32 (single warp), each thread now owns 4 d-lanes (`head_dim/32`), so the entire dot-product reduce fits in one `warp_reduce_sum` — every `__syncthreads()` is gone and shmem usage drops to 0 B. (b) 64-bit vectorized KV loads via `uint2` + `__half22float2`: one warp-wide K load is now 256 B in a single `LDG.E.64` instruction (vs v2's four 64 B warp loads). Output write also vectorized to one `STG.E.64`. Trade-off accepted: per-SM occupancy drops 48→16 warps (full→33%) because each block has only 1 warp, but the vec throughput + sync removal more than compensates. REG: 33 (v2: 28). SHARED: 0 B (v2: 256). Now 1.91× faster than PyTorch SDPA. Remaining bandwidth headroom: 189/1008 ≈ 19% of peak HBM, so v4/v5 still have room. ncu pending. |
| + split-K (FlashDecoding, K_SPLIT=8) | f904aae (explored, then reverted) | 0.802 ms | 167 GB/s | **0.89× (regression)** | Two-kernel design: stage 1 (grid `batch×heads×K_SPLIT`, each block runs the v3 body over `j ∈ [s·chunk, (s+1)·chunk)` and writes unnormalized `o_acc`, `(m, l)` to scratch); stage 2 (grid `batch×heads`, merges K_SPLIT partials via `m_final = max_s m_s`, `l_final = Σ_s l_s·exp(m_s−m_final)`, `o_final = (Σ_s o_s·exp(m_s−m_final))/l_final`). Tried K_SPLIT ∈ {2, 4, 8, 16}; 8 and 16 were best, 4 was 1.02 ms. Cross-batch sweep showed v4 is roughly *flat* at 0.67–0.80 ms across batch ∈ {1, 2, 4, 8}, vs v3's 0.44–0.72 ms — i.e. ~250 µs of fixed overhead that doesn't unlock new parallelism. Diagnosis: v3 wasn't actually grid-limited at our workload — only ~11 SMs busy at batch=1 but those SMs were already at the HBM/L2-throughput limit, so adding 8× more SMs via split-K just made more SMs share the same constrained bandwidth. Consolidating 3 `cudaMallocAsync` → 1 didn't help, so launch/alloc overhead isn't the dominant cost either; the combine kernel + scratch traffic + 2nd launch ≈ 250 µs together. Split-K is the right tool for *truly* grid-undersized cases (batch=1 with `n_heads=8`-ish where v3 fills <8 SMs); ours had `batch×n_heads=256` blocks already and bandwidth was the ceiling. **Conclusion: not landed on `main`; v3 remains the current best.** |
| + cp.async double-buffer (v5) | 78a28ff (explored, then reverted) | 0.760 ms | 177 GB/s | **0.94× (regression)** | Two-stage pipeline via `__pipeline_memcpy_async` (cp.async.ca.shared.global, 8 B per thread). Prime tile 0 → shmem slot 0; in the loop, prefetch tile j+1 → slot ((j+1)&1), `__pipeline_wait_prior(1)`, read tile j from its slot. Each thread cp.asyncs only its own 8-byte slot, so no `__syncwarp` needed. Cross-batch sweep: v5 0.51 / 0.51 / 0.51 / 0.76 ms vs v3 0.44 / 0.44 / 0.44 / 0.71 ms at batch ∈ {1, 2, 4, 8} — uniformly ~50 µs slower. Diagnosis: cp.async only writes to shmem, so we paid an extra per-iter shmem-write + shmem-read hop (~10 cycles/iter × 4096 iters ≈ tens of µs per block) on top of v3's direct global → register pattern. nvcc was *already* pipelining v3's loads through ordinary load latency hiding; the explicit cp.async pipeline didn't expose meaningful new overlap. cp.async also bypasses L1 (goes L2-only) — neutral here since K+V per head (~2 MB) doesn't fit L1 (128 KB) anyway, but it's a cost when the access pattern would benefit from L1. Did **not** try deeper pipelining (NUM_STAGES=3/4) because the dominant cost is the depth-independent shmem hop. **Conclusion: not landed on `main`; v3 remains the current best.** |

Correctness: v0 max |abs diff| vs fp32 reference = 6.1e-5 (well under the
2e-2 gate). `tests/test_attention.py` green at batch ∈ {1, 8} ×
seqlen_kv ∈ {128, 2048}.

**vs SOTA:** `flash_attn` decode = _TBD_ µs. Gap = _TBD_%. Explanation: _TBD_.

## Track 2 — KV-cache compression

Microbench workload: same as Track 1 — Llama 3 8B head config (`n_heads=32,
n_kv_heads=8, head_dim=128`), `batch=8, seqlen_kv=4096`, fp16 inputs.
"Memory" = on-device steady-state KV-cache size including scale overhead.
"Kernel latency" measured by `benchmarks/bench_kv_cache.py`.
"Perplexity Δ" measured by `scripts/eval_perplexity.py` on Llama 3.1 8B
Instruct over WikiText-2 test (131,008 tokens, 64 chunks × 2048 tokens),
fp16 baseline ppl = 7.055.

| Variant | Commit | Memory | Kernel latency | Perplexity Δ | Notes |
|---------|--------|--------|----------------|--------------|-------|
| FP16 KV (baseline) | _Track 1 v3_ | 128 MiB · 1.00× | 0.713 ms | 0.0 | v3 fused decode attention. 1.91× faster than PyTorch SDPA. |
| INT8 KV per-token + fused dequant | _HEAD_ | 65 MiB · **0.51×** (63 MiB saved) | 0.713 ms (tied with v3) | **+0.0008** (+0.01%) — essentially lossless | Symmetric per-token quantization (one fp16 scale per token, shared across head_dim). Kernel: same v3 body, ints loaded as `LDG.E.32` (4 bytes/thread vs v3's 8), scales folded out of per-lane dequant (`partial · k_scale` *after* `warp_reduce_sum`; `p_j_scaled = p_j · v_scale` folded into the V FMA). Latency parity with v3 despite half the KV bytes — kernel is dependency-chain-bound (warp reduce → softmax → FMA), not bandwidth-bound (96/1008 GB/s ≈ 9.5% of peak HBM). Accuracy: max abs diff vs fp16 reference 1.1e-3, mean rel err 2.5% (kernel-level on random gaussians); model-level Δppl 0.0008 on WikiText-2 — well under the 0.2 threshold. Quantize one-shot cost: 0.124 ms per K (full cache); in serving this amortises to per-appended-token cost. |
| INT4 KIVI (per-channel K, per-token V) | _HEAD_ | 34.5 MiB · **0.27×** (93.5 MiB saved) | **0.554 ms** (**1.29× over v3**) | **+0.196** (+2.78%) — under 0.5 target | KIVI layout: K quantized per-channel with group_size=32 tokens along the seqlen axis, V quantized per-token. Packed 4-bit storage (`int8` byte = 2 nibbles). The K scales are now loaded **once per group** (every 32 j-iters), held in registers, and pre-folded into q via `q_scaled[d] = q[d] · k_scale[g, d]`. Inner loop is therefore 4 multiplies on int values + 4 fmas with int V values — no per-iter K scale work. Latency wins where INT8 tied: the smaller per-iter load (`LDG.E.16` for K_q + V_q, 2 B/thread each) plus the saved per-iter K scale load make the inner loop denser and the dependency chain effectively shorter. Accuracy on random gaussians: max abs diff 2.3e-2, mean rel err 42% (worst case on uniform noise). Model-level: Δppl 0.196 — **2.36× better than naive INT4 per-token K (Δppl 0.462)**, direct experimental confirmation that K's per-channel outliers need their own scales. Quantize one-shot costs: K 0.07 ms, V 0.12 ms (amortizes per-token in serving). |

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
