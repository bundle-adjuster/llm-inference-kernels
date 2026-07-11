# Results Log

> ## ⚠ Correction (Phase 7) + ✅ Fix (Phase 8) — read before trusting any "vs SDPA" or "vs HF" number below
>
> The PyTorch SDPA baseline used throughout Phases 1–4 was handed a
> **4×-expanded GQA key/value cache** (`reference/attention_ref.py:94`
> `_expand_gqa`; transformers' `repeat_kv` in the e2e path). SDPA's native GQA
> path is 4–12× faster than the expanded one. Consequences:
>
> - Phase 1's **"1.91× over PyTorch SDPA" was retired.** Measured against
>   `F.scaled_dot_product_attention(..., enable_gqa=True)`, the v3 kernel is
>   **4.55× slower** on its own reference workload — it was occupancy-bound.
> - **✅ Phase 8 fixed it.** v6 is FlashDecoding split-K on multi-warp blocks (the
>   fix Phase 7 identified). On the reference workload it runs **155.6 µs vs fair
>   SDPA's 157.3 µs — a 4.59× speedup over v3**, at ~82% of peak HBM. The custom
>   kernel now reaches **parity with PyTorch SDPA** on the HBM-bound shapes
>   (1.01–1.02×, a 1–2% margin — the noise floor on unlocked clocks; the robust
>   win is 4.59× over v3 and 82% vs 18% of peak) and edges the fair baseline
>   end-to-end (538.6 vs 533.1 tok/s). See
>   [`../06-attention-splitk-journey.md`](../06-attention-splitk-journey.md).
> - Phase 1's **v4 split-K revert was misdiagnosed.** It blamed a bandwidth
>   ceiling; flash's split-K kernel reaches 81% of peak HBM where v3 reaches 18%.
>   v3 was occupancy-bound — and Phase 8's v6 proves it by fixing occupancy and
>   reaching parity with flash.
> - Phase 4b's **1.55× e2e was `repeat_kv` removal, not the INT4 kernel.** 4b is
>   a memory result. Plain fp16 `enable_gqa` scores ~533 tok/s vs 4b's 521.7.
> - **vLLM's 2.09× over HF is framework tax, not kernels.** Against a fair
>   baseline the gap is 1.31×, and vLLM's GEMMs run no faster than ours. Phase 8's
>   attention win does not close it — the lever is W4A16.
>
> Full analysis, method, and numbers:
> [`../05-baseline-correction-journey.md`](../05-baseline-correction-journey.md)
> (correction) and [`../06-attention-splitk-journey.md`](../06-attention-splitk-journey.md)
> (fix). Harness: [`../../benchmarks/bench_decode_step.py`](../../benchmarks/bench_decode_step.py).
> Original numbers below are preserved with their handicap named, not deleted.

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

**Phase 7 correction.** That 1.98× is almost entirely transformers' `repeat_kv`
GQA expansion, not a kernel difference. Removing it (one line: SDPA's
`enable_gqa=True`) takes HF to **535.2 tok/s / 1.55×** with no custom CUDA, and
cuts vLLM's remaining lead to **1.31×**. Both engines' projection GEMMs stream
the same 15.01 GB of fp16 weights per decode step, ours at 740 GB/s and vLLM's
at an implied ~715 GB/s. Use `gqa` — not `vanilla` — as the denominator for any
kernel claim. See
[`../05-baseline-correction-journey.md`](../05-baseline-correction-journey.md).

## Track 1 — Fused attention

Microbench workload: Llama 3 8B head config (`n_heads=32, n_kv_heads=8,
head_dim=128`), `batch=8, seqlen_kv=4096`, fp16. Measured by
`benchmarks/bench_attention.py` (CUDA events, 25 warmup + 100 timed).
Achieved BW = `(|K|+|V|) / median_latency`; each tensor is 64 MB so 128 MB
streamed per call. For reference on this workload: PyTorch eager 3.77 ms,
PyTorch SDPA 1.36 ms. Clocks not yet locked (TODO before `ncu` runs).

> **⚠ The "PyTorch SDPA 1.36 ms" reference is a handicapped baseline; every
> "vs SDPA" / "faster than PyTorch SDPA" phrase in the table below is retired.**
> `reference/attention_ref.py:94` expands GQA (8 kv heads → 32) before calling
> SDPA. Given the un-expanded cache via `enable_gqa=True`, SDPA does the same
> work in **157.3 µs**, so those ratios are inflated ~8.7× and v3 actually *loses*
> to fair SDPA by 4.55×. The CUDA-vs-CUDA ratios (vs v0/prev) are unaffected and
> remain valid — read only those. **✅ Phase 8 then built the kernel that does
> beat fair SDPA: v6 split-K, 1.01× on the reference workload — see the "vs SOTA"
> table below and [`../06-attention-splitk-journey.md`](../06-attention-splitk-journey.md).**

| Step | Commit | Latency | Achieved BW | Speedup vs prev | Cause (ncu metric) |
|------|--------|---------|-------------|-----------------|--------------------|
| v0 naive (two-pass softmax) | 46930c2 | 1.669 ms | 80 GB/s | — (baseline) | per-`j` block reductions in phase 1 serialize compute; KV is read in two passes (K in phase 1, V in phase 3); GQA reads not de-duplicated across query heads sharing a kv head. ncu profile pending (lock clocks first). |
| + online softmax (naive port) | 46ae1ea | 2.078 ms | 65 GB/s | **0.80× (regression)** | Textbook Milakov–Gimelshein recurrence per-thread. The V load is issued at point of use, *inside* the per-`j` `__syncthreads()` barriers — so its latency can't hide behind the K reduction. v0 doesn't suffer this because its phase 3 V-loop is sync-free. (Also suspected at the time: ~128× redundant `__expf` calls, since the scalar `alpha`/`p_j` recurrence is duplicated across threads — but moving it to lane 0 + shmem broadcast made things *worse* (2.26 ms), confirming SIMT-redundant ALU work is essentially free.) |
| + V-load prefetch | ad9c57f | 1.637 ms | 82 GB/s | 1.27× over naive port; 1.02× over v0 | Issue `v_j = V[j, tid]` at the *top* of the iteration, alongside `k_j`. The load is non-blocking; the value isn't consumed until after both `__syncthreads()` and the softmax update, so the V latency hides behind ~all of that work. nvcc would not hoist the load above the syncs on its own. Now consistent with theory: single-pass + no shmem score buffer + sync-overlapped V latency. ncu pending. |
| v2 (single-sync block reduce) | db6ab0b | 1.069 ms | 126 GB/s | **1.53× over v1-prefetch; 1.56× over v0** | Two changes drop the per-`j` sync count from 2 to 1. (a) All warps redundantly do the final cross-warp reduce — read `reduce_smem[buf][0..n_warps]`, mask, `warp_reduce_sum`. Every thread arrives at the same `s_j` via warp shuffle, no shmem broadcast. (b) Double-buffer `reduce_smem` on `j & 1` so iter j+1's write never races iter j's still-in-flight reads. The hazard between iter j's read and iter j+2's write (same slot) is gated by iter j+1's sync. The speedup is larger than the saved sync alone would predict (~150µs predicted vs 570µs measured) — removing the `s_bcast` shmem hop also lets `s_j` flow directly from the shfl tree into the softmax FMAs. Now beating PyTorch SDPA (1.36 ms) by 1.27×. ncu pending. _(Phase 7: that "1.36 ms SDPA" is the handicapped GQA-expanded baseline — fair SDPA is 157.3 µs, so v2 loses to it; the 1.53×/1.56× CUDA-vs-CUDA ratios stand. Phase 8 v6 is what beats fair SDPA — see docs/06.)_ |
| v3 (vectorized KV loads) | ccdb6df | 0.713 ms | 189 GB/s | **1.50× over v2; 2.34× over v0** | Two changes: (a) block shrinks 128→32 (single warp), each thread now owns 4 d-lanes (`head_dim/32`), so the entire dot-product reduce fits in one `warp_reduce_sum` — every `__syncthreads()` is gone and shmem usage drops to 0 B. (b) 64-bit vectorized KV loads via `uint2` + `__half22float2`: one warp-wide K load is now 256 B in a single `LDG.E.64` instruction (vs v2's four 64 B warp loads). Output write also vectorized to one `STG.E.64`. Trade-off accepted: per-SM occupancy drops 48→16 warps (full→33%) because each block has only 1 warp, but the vec throughput + sync removal more than compensates. REG: 33 (v2: 28). SHARED: 0 B (v2: 256). Now 1.91× faster than PyTorch SDPA. Remaining bandwidth headroom: 189/1008 ≈ 19% of peak HBM, so v4/v5 still have room. ncu pending. _(Phase 7: "1.91× faster than SDPA" is retired — vs fair `enable_gqa` SDPA (157.3 µs) v3 is 4.55× **slower** at 18% of peak HBM; it was occupancy-bound (~2 of 128 SMs), not bandwidth-bound, so the "still have room" reading was wrong. Phase 8 v6 fixed occupancy and beats fair SDPA at 1.01× / 82% peak — see docs/06. CUDA-vs-CUDA 1.50×/2.34× stand.)_ |
| + split-K (FlashDecoding, K_SPLIT=8) | f904aae (explored, then reverted) | 0.802 ms | 167 GB/s | **0.89× (regression)** | Two-kernel design: stage 1 (grid `batch×heads×K_SPLIT`, each block runs the v3 body over `j ∈ [s·chunk, (s+1)·chunk)` and writes unnormalized `o_acc`, `(m, l)` to scratch); stage 2 (grid `batch×heads`, merges K_SPLIT partials via `m_final = max_s m_s`, `l_final = Σ_s l_s·exp(m_s−m_final)`, `o_final = (Σ_s o_s·exp(m_s−m_final))/l_final`). Tried K_SPLIT ∈ {2, 4, 8, 16}; 8 and 16 were best, 4 was 1.02 ms. Cross-batch sweep showed v4 is roughly *flat* at 0.67–0.80 ms across batch ∈ {1, 2, 4, 8}, vs v3's 0.44–0.72 ms — i.e. ~250 µs of fixed overhead that doesn't unlock new parallelism. Diagnosis: v3 wasn't actually grid-limited at our workload — only ~11 SMs busy at batch=1 but those SMs were already at the HBM/L2-throughput limit, so adding 8× more SMs via split-K just made more SMs share the same constrained bandwidth. Consolidating 3 `cudaMallocAsync` → 1 didn't help, so launch/alloc overhead isn't the dominant cost either; the combine kernel + scratch traffic + 2nd launch ≈ 250 µs together. Split-K is the right tool for *truly* grid-undersized cases (batch=1 with `n_heads=8`-ish where v3 fills <8 SMs); ours had `batch×n_heads=256` blocks already and bandwidth was the ceiling. **Conclusion: not landed on `main`; v3 remains the current best.** _(Phase 7/8: this "bandwidth was the ceiling" diagnosis is wrong — v3 reached only 18% of peak HBM, so bandwidth was not the limit; v3 was occupancy-bound (its single-warp blocks fill ~2 of 128 SMs). This split-K regressed only because it kept v3's single-warp body. Phase 8's v6 does split-K on 4-warp blocks and reaches 82% of peak, beating fair SDPA — see docs/06.)_ |
| + cp.async double-buffer (v5) | 78a28ff (explored, then reverted) | 0.760 ms | 177 GB/s | **0.94× (regression)** | Two-stage pipeline via `__pipeline_memcpy_async` (cp.async.ca.shared.global, 8 B per thread). Prime tile 0 → shmem slot 0; in the loop, prefetch tile j+1 → slot ((j+1)&1), `__pipeline_wait_prior(1)`, read tile j from its slot. Each thread cp.asyncs only its own 8-byte slot, so no `__syncwarp` needed. Cross-batch sweep: v5 0.51 / 0.51 / 0.51 / 0.76 ms vs v3 0.44 / 0.44 / 0.44 / 0.71 ms at batch ∈ {1, 2, 4, 8} — uniformly ~50 µs slower. Diagnosis: cp.async only writes to shmem, so we paid an extra per-iter shmem-write + shmem-read hop (~10 cycles/iter × 4096 iters ≈ tens of µs per block) on top of v3's direct global → register pattern. nvcc was *already* pipelining v3's loads through ordinary load latency hiding; the explicit cp.async pipeline didn't expose meaningful new overlap. cp.async also bypasses L1 (goes L2-only) — neutral here since K+V per head (~2 MB) doesn't fit L1 (128 KB) anyway, but it's a cost when the access pattern would benefit from L1. Did **not** try deeper pipelining (NUM_STAGES=3/4) because the dominant cost is the depth-independent shmem hop. **Conclusion: not landed on `main`; v3 remains the current best.** |

Correctness: v0 max |abs diff| vs fp32 reference = 6.1e-5 (well under the
2e-2 gate). `tests/test_attention.py` green at batch ∈ {1, 8} ×
seqlen_kv ∈ {128, 2048}.

**vs SOTA** (filled in at Phase 7; v6 landed Phase 8; `bench_decode_step.py --part kernel`):

| Workload | SDPA `enable_gqa` (flash split-KV) | v3 (retired) | **v6 split-K** | v6 vs SDPA |
|---|---|---|---|---|
| `batch=8, kv_len=4096` (Phase 1 reference) | **157.3 µs** (~81% peak) | 713.7 µs · 0.22× | **155.6 µs** (~82% peak) | **1.01× — parity** |
| `batch=16, kv_len=2048` | **157.7 µs** | 389.8 µs · 0.40× | **154.0 µs** | **1.02×** |
| `batch=16, kv_len=768` (L2-resident) | **37.5 µs** | 95.2 µs · 0.40× | 53.0 µs | 0.71× |
| `batch=8, kv_len=1024` (L2-resident) | **28.7 µs** | 113.4 µs · 0.25× | 34.8 µs | 0.82× |

**Explanation (Phase 7 diagnosis, Phase 8 fix).** torch dispatches
`flash_fwd_splitkv_kernel` — FlashDecoding, split-K over the sequence — reaching
**81% of peak HBM**. v3's single-warp block reached only 18%: it was
**occupancy-bound, not bandwidth-bound** (it fills ~2 of 128 SMs at the reference
workload). The v4 step misattributed its split-K regression to a bandwidth
ceiling flash exceeds by 4.5× — the ceiling did not exist. **Phase 8's v6 does
split-K right:** many blocks per (batch, head) to fill the SMs, 4 warps/block for
full occupancy, and a 4-deep unrolled load loop for memory-level parallelism. It
reaches ~82% of peak and **matches fair SDPA (1.01–1.02× — parity within the
noise floor) on the HBM-bound shapes**,
a 4.59× speedup over v3. It still trails flash on the small **L2-resident** shapes
(0.69–0.82×), where the kernel is L2-bound rather than HBM-bound — the honest
remaining gap. See [`../06-attention-splitk-journey.md`](../06-attention-splitk-journey.md).

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
| FP16 KV (baseline) | _Track 1 v3_ | 128 MiB · 1.00× | 0.713 ms | 0.0 | v3 fused decode attention. (Phase 7: the "1.91× faster than PyTorch SDPA" that stood here is retired — v3 actually loses to fair `enable_gqa` SDPA by 4.55×; see the correction banner. ✅ Phase 8's v6 split-K is the kernel that beats fair SDPA (1.01× on this workload — docs/06). The Track 2 memory and accuracy results are unaffected; the latency column is CUDA-vs-CUDA against v3.) |
| INT8 KV per-token + fused dequant | _HEAD_ | 65 MiB · **0.51×** (63 MiB saved) | 0.713 ms (tied with v3) | **+0.0008** (+0.01%) — essentially lossless | Symmetric per-token quantization (one fp16 scale per token, shared across head_dim). Kernel: same v3 body, ints loaded as `LDG.E.32` (4 bytes/thread vs v3's 8), scales folded out of per-lane dequant (`partial · k_scale` *after* `warp_reduce_sum`; `p_j_scaled = p_j · v_scale` folded into the V FMA). Latency parity with v3 despite half the KV bytes — kernel is dependency-chain-bound (warp reduce → softmax → FMA), not bandwidth-bound (96/1008 GB/s ≈ 9.5% of peak HBM). Accuracy: max abs diff vs fp16 reference 1.1e-3, mean rel err 2.5% (kernel-level on random gaussians); model-level Δppl 0.0008 on WikiText-2 — well under the 0.2 threshold. Quantize one-shot cost: 0.124 ms per K (full cache); in serving this amortises to per-appended-token cost. |
| INT4 KIVI (per-channel K, per-token V) | _HEAD_ | 34.5 MiB · **0.27×** (93.5 MiB saved) | **0.554 ms** (**1.29× over v3**) | **+0.196** (+2.78%) — under 0.5 target | KIVI layout: K quantized per-channel with group_size=32 tokens along the seqlen axis, V quantized per-token. Packed 4-bit storage (`int8` byte = 2 nibbles). The K scales are now loaded **once per group** (every 32 j-iters), held in registers, and pre-folded into q via `q_scaled[d] = q[d] · k_scale[g, d]`. Inner loop is therefore 4 multiplies on int values + 4 fmas with int V values — no per-iter K scale work. Latency wins where INT8 tied: the smaller per-iter load (`LDG.E.16` for K_q + V_q, 2 B/thread each) plus the saved per-iter K scale load make the inner loop denser and the dependency chain effectively shorter. Accuracy on random gaussians: max abs diff 2.3e-2, mean rel err 42% (worst case on uniform noise). Model-level: Δppl 0.196 — **2.36× better than naive INT4 per-token K (Δppl 0.462)**, direct experimental confirmation that K's per-channel outliers need their own scales. Quantize one-shot costs: K 0.07 ms, V 0.12 ms (amortizes per-token in serving). |

## Track 3 — Quantized matmul (W4A16)

Microbench workload: Llama 3 8B linear-layer shapes. Symmetric INT4
weights with groupwise scales along K (group_size=128). Activations
fp16. Measured by an ad-hoc bench against `torch.matmul` (cuBLAS fp16).

`Speedup vs fp16` is `cuBLAS_latency / w4a16_latency` — values > 1 mean
this kernel beats fp16 cuBLAS.

### Phase 3b — naive W4A16 GEMM (one warp per output tile of 32 columns)

| Shape (K, N)   | M  | Commit | cuBLAS fp16 | w4a16 naive | Speedup |
|----------------|---:|--------|------------:|------------:|--------:|
| 4096 × 4096    |  1 | _HEAD_ | 0.047 ms    | 0.088 ms    | 0.53× |
| **4096 × 14336** (MLP up/gate) | **1** | _HEAD_ | **0.133 ms** | **0.084 ms** | **1.59×** |
| 14336 × 4096 (MLP down) |  1 | _HEAD_ | 0.133 ms | 0.284 ms | 0.47× |
| (any)          |  8 | _HEAD_ | ~M=1 |   scales linearly with M | ≤ 0.21× |
| (any)          | 32 | _HEAD_ | ~M=1 |   scales linearly with M | ≤ 0.05× |

Correctness vs reference: max abs diff at rtol/atol = 2e-2 across all
6 (shape, M) configs in `tests/test_quant.py::test_cuda_w4a16_gemm_matches_reference`.

**Headline**: at the canonical decode shape (M=1, K=4096, N=14336), the
naive kernel **beats fp16 cuBLAS by 1.59×**. The W4A16 thesis holds —
when grid parallelism is enough (large N) and the bottleneck is weight
HBM traffic (small M), 4× less weight bytes ≈ proportional latency win.

**What the naive doesn't handle**, deferred to Phase 3c:
- Small N (=4096): only 128 blocks of 32 columns — under-fills the
  4090's 128 SMs.
- M > 1: kernel iterates M sequentially, reloading weights per row.
  cuBLAS amortises by batching M into a GEMM.
- Per-thread bookkeeping in long-K cases (K=14336): no vectorisation,
  no `act` shmem caching. L1 catches the act reuse across the 32
  threads in a block but not across blocks.

### Phase 3c — decode-optimized W4A16 GEMM (M=1 fast path)

Two changes vs 3b:

1. **Multi-warp block + K-split.** Block grows from 1 warp to 4 warps
   (128 threads). The 32 columns are still owned by one warp's 32
   lanes — but K is split across the 4 warps: each warp processes
   `K/4` of the reduction for all 32 columns. After the K loop, a
   tiny shmem combine sums the 4 per-warp partials per column. This
   gives both per-block speedup (more compute throughput per block)
   and better grid coverage at small N (4× the warp count per kernel).
2. **`act` in shared memory.** The K activation vector is loaded once
   per block into shmem at kernel start (cooperative across the 128
   threads), then read from shmem in the inner loop. Frees L1 for the
   weight traffic.

Fast path applies only at M=1. M>1 falls back to the 3b naive kernel
via the launcher dispatch — 3c is decode-specialized; batched-decode
M>1 is integration concern (Phase 4).

| Shape (K, N)   | M | Commit | cuBLAS fp16 | w4a16 decode | Speedup |
|----------------|--:|--------|------------:|-------------:|--------:|
| 4096 × 4096 (attn QKV / O)    | 1 | _HEAD_ | 0.047 ms | **0.016 ms** | **2.88×** |
| **4096 × 14336 (MLP up/gate)**| 1 | _HEAD_ | 0.134 ms | **0.019 ms** | **6.97×** |
| 14336 × 4096 (MLP down)        | 1 | _HEAD_ | 0.133 ms | **0.045 ms** | **2.96×** |

All three M=1 Llama 3 8B linear-layer shapes clear the Phase 3 *Target*
(2–3× over fp16 cuBLAS) at M=1. MLP up/gate clears it by ~3.5×.

**Why each shape improved over 3b** (per Phase 3 journey notes):

- **4096 × 4096** (3b: 0.53×, 3c: 2.88× — **5.4× improvement**): 3b had
  only 128 single-warp blocks → ~1 warp per SM, severe under-occupation.
  3c's 4-warp blocks give 512 warps total — 4 warps/SM, meaningful for
  latency hiding.
- **4096 × 14336** (3b: 1.59×, 3c: 6.97× — **4.4× improvement**): 3b
  already won via grid coverage. 3c additionally cuts per-thread K work
  4× and adds act caching.
- **14336 × 4096** (3b: 0.47×, 3c: 2.96× — **6.3× improvement**): 3b's
  worst loss was here — long K serialised through one thread. K-splitting
  brought per-thread K from 14336 to 3584.

**vs Marlin (SOTA)**: not yet measured. The 6.97× win on MLP up/gate
puts us in the right neighbourhood for the docs/03 *Target* of "within
25% of Marlin"; full comparison deferred to a follow-up.

M > 1: still falls back to the naive kernel. The bench numbers above
show the gap (worsens linearly with M); a batched-decode-optimized
variant would need M-fast inner loops and is a Phase 4 concern.

## End-to-end (Phase 4)

Goal of Phase 4: take the three kernels built in Phase 1–3 and plug them
back into the actual Llama 3.1 8B Instruct model via HF monkeypatch, one at
a time, and measure the accuracy / latency / memory tradeoff at each step.

**Eval bar** (the GPTQ/AWQ/KIVI paper convention, plus our own end-to-end metrics):

- **Standard accuracy** (`scripts/run_lm_eval.py`, lm-evaluation-harness):
  MMLU 5-shot, HellaSwag 0-shot (acc_norm), ARC-Challenge 25-shot (acc_norm).
- **End-to-end** (`scripts/run_e2e_eval.py`): WikiText-2 PPL (64 chunks × 2048
  tok), greedy-token match rate on 10 fixed prompts vs vanilla reference,
  decode tokens/sec on the LOCKED workload (batch=16, prompt=512, gen=512),
  peak VRAM.

Per-config JSON outputs in `lm_eval/` and `e2e_eval/`. The vanilla greedy
reference (10 prompts × 64 new tokens) is committed at
`e2e_eval/vanilla_reference_outputs.json` — 4a/4b/4c match against it.

> **⚠ Phase 7:** the `vs HF` column below is measured against `vanilla HF`, a
> baseline carrying ~14 ms/step of `repeat_kv` + `cat` overhead. The honest
> denominator is the `gqa` config (**535.2 tok/s**, stock PyTorch with
> `enable_gqa=True`, no custom CUDA). Read `vs fair` for the kernel's actual
> contribution. The `vs vLLM` column compares a research harness to a serving
> engine and should not be read as a kernel comparison at all.

| Config | Branch | Tokens/sec | vs HF | vs fair | vs vLLM | Peak VRAM | PPL | greedy | MMLU | HellaSwag | ARC-C |
|--------|--------|-----------|------:|--------:|--------:|----------:|----:|-------:|-----:|----------:|------:|
| **vanilla HF** (Phase 4 baseline) | `phase4-eval-prep` | 335.8 | 1.00× | 0.63× | 0.48× | 18.50 GB | 7.055 | 1.0000 | 68.32% | 79.51% | 60.84% |
| **`gqa` fair baseline** (Phase 7) | `bench_decode_step.py` | **535.2** | 1.59× | **1.00×** | 0.76× | n/m | — | — | — | — | — |
| + fused attention (4a) | `phase4-attention` | 344.1 | 1.025× | 0.64× | 0.49× | 18.57 GB | 7.055 | 1.0000 | 68.32% | 79.51% | 60.84% |
| + INT4 KIVI KV cache (4b) | `phase4-kv-int4` | 521.7 | 1.55× | **0.97×** | 0.74× | 18.41 GB | 7.256 | 0.5047 | 67.29% | 79.07% | 61.43% |
| + W4A16 weights (4c, Phase 5 kernel) | `phase4-w4a16` + `phase5-batched-w4a16` | 199.9 | 0.60× | 0.37× | 0.28× | 9.05 GB | 8.087 | 0.2672 | 62.40% | 77.51% | 55.72% |
| + W4A16 weights (4c) | `phase4-w4a16` | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| vanilla vLLM (ref) | _ | 703.2 | 2.09× | 1.31× | 1.00× | n/a | n/a | n/a | n/a | n/a | n/a |

**Reading the `vs fair` column.** 4b lands at **0.97×** — the INT4 KIVI kernel is
marginally *slower* than doing nothing but letting SDPA handle GQA natively, and
it costs Δppl +0.196 and a greedy match rate of 0.5047 to get there. 4b's
headline 1.55× was `repeat_kv` removal (its forward replacement skips it), not
the kernel. **4b is a memory result, not a speed result.** Likewise 4a's 1.025×
is a slower kernel partially offsetting an expansion it never avoided (it hooks
`F.sdpa`, which HF calls *after* `repeat_kv`).

Caveat: 4a/4b/4c tok/s come from the Phase 4 eval run (vanilla 335.8) and the
`gqa` number from the Phase 7 run (vanilla 344.8) — a ~2.7% cross-run offset on
unlocked clocks. The `vs fair` ratios are good to about ±3% and the conclusions
do not turn on that margin. A single combined run with locked clocks is the fix.

**4-prep notes** (vanilla baseline, this commit):
- HF tokens/sec is slightly below the Phase 0 number (354.6) — same setup,
  different measurement run; gap is within run-to-run noise on unlocked
  clocks. Phase 4 deltas are measured against this 335.8 reference for
  apples-to-apples (all four configs share an eval run).
- Llama 3.1 8B Instruct accuracy on MMLU/HellaSwag/ARC-C agrees with
  Meta's reported numbers (69.4 / 80.4 / 60.3) within ~1pp — gives us
  confidence in the eval setup before any kernel goes in.
- vanilla vLLM tokens/sec is the Phase 0 number; accuracy eval through
  vLLM is deferred (no clean HFLM hook), and is not strictly needed for
  the Phase 4 narrative — our kernels integrate into HF, not vLLM.

**4a notes** (attention kernel integration):
- Patch: rebind `F.scaled_dot_product_attention` so `q_len == 1` (decode)
  dispatches to our Phase 1 v3 `decode_attention`; `q_len > 1` (prefill)
  falls through to the original SDPA. Un-expands HF's `repeat_kv` via
  `K[:, ::n_rep].contiguous()` so the kernel reads only `n_kv_heads = 8`
  rows of KV, not the post-expand 32. Source: `integration/attention_patch.py`.
- Accuracy is **bit-identical** to vanilla on every prefill-based metric
  (PPL, MMLU, HellaSwag, ARC-C all match to the last reported digit) —
  because prefill falls through to the original SDPA. `greedy_match_rate
  = 1.0000` confirms decode is bit-perfect too: our kernel and SDPA produce
  the same token under greedy sampling on all 10 × 64 = 640 generated
  tokens. The kernel is a pure performance change, no accuracy tradeoff.
- **⚠ Phase 7 correction — the estimate in this bullet is wrong.** It computes
  attention traffic on the *un-expanded* 8-head cache (~50 MB/layer), but HF
  hands SDPA the `repeat_kv`-expanded 32-head cache (~143 MB/layer). Measured,
  attention is **7.54 ms** and the expansion that feeds it another **6.7 ms** of
  the 37–44 ms step — together ~35%, not ~4%. The conclusion drawn from the bad
  estimate ("big wins come from 4c, not 4a") is backwards: removing `repeat_kv`
  is the single largest e2e win available, worth 1.55×. It just isn't a kernel
  win. The original reasoning is preserved below.
- Tokens/sec gain is small (1.025×) because attention is only ~4% of decode
  time at this workload — projection GEMMs dominate. At batch=16, avg
  kv_len ≈ 768, head_dim=128, n_kv_heads=8, attention reads ~50 MB/layer
  × 32 layers / ~1 TB/s ≈ 1.6 ms of the ~41 ms decode step. Doubling
  attention speed saves ~0.8 ms / 41 ms ≈ 2%. The Phase 1 microbench's
  1.91× was real but on a workload where attention was the *whole* thing
  (no GEMMs around it). Big wins on this e2e workload come from 4c (W4A16
  on projections), not 4a.
  - **⚠ Phase 7/8 — the "1.91× was real" clause is wrong.** That microbench
    number was vs the GQA-expanded SDPA baseline; vs fair `enable_gqa` SDPA the
    v3 kernel *loses* (0.22×), so there was never a real 1.91× attention win to
    plug in — 4a's own 1.025× reflects a slower kernel. Phase 8's v6 split-K is
    the first kernel that genuinely beats fair SDPA (1.01× kernel, 538.6 vs the
    533.1 tok/s fair e2e baseline). See docs/06-attention-splitk-journey.md.
- Peak VRAM up 0.07 GB = the un-expansion buffer (~66 MB). Negligible
  vs total. Avoiding it would require subclassing `LlamaSdpaAttention`
  to skip `repeat_kv` — deferred (not worth the code surface for 0.4%).

**4b notes** (KV-cache compression integration):
- Two patches working together:
  - `Int4KIVICache` (`integration/kv_int4_cache.py`): a real `transformers.Cache`
    subclass that stores K as packed INT4 + per-channel groupwise scales
    (group=32) and V as packed INT4 + per-token scales — the KIVI scheme
    proven in Phase 2c. A small fp16 residual buffer holds tokens not yet in
    a full group; once it reaches `group_size`, the chunk is quantized into
    the packed storage.
  - `patched_int4_decode_attention` (in `integration/attention_patch.py`):
    replaces `LlamaSdpaAttention.forward` so at decode (q_len=1) it bypasses
    `repeat_kv` + SDPA entirely, calling our `decode_attention_int4` kernel
    on the cache's packed tensors directly. Prefill (q_len>1) falls back to
    the original forward — `cache.update()` returns dequantized fp16 K/V so
    the SDPA path sees KIVI-noisy attention inputs.
  - For lm-evaluation-harness (which doesn't pass `past_key_values` through
    `HFLM`), a separate context (`patched_kivi_int4_sdpa`) rebinds `F.sdpa`
    to do the same KIVI quantize-and-dequantize on K/V before the original
    SDPA call. Identical math to the cache path → identical noise pattern,
    so the lm-eval numbers reflect the same KIVI accuracy hit the cache
    path produces.
- Accuracy hit is small and in line with KIVI literature: **MMLU -1.03 pp**
  (68.32 → 67.29), **HellaSwag -0.44 pp**, **ARC-C +0.59 pp** (noise; small
  set), **PPL +0.20** (closely matches Phase 2c's kernel-level +0.196 — a
  validation that the integration faithfully reproduces the kernel's noise
  characteristics). `greedy_match_rate` drops to 0.50 because the small
  per-token logit noise from KIVI flips argmax choices a few tokens in,
  and greedy decoding compounds those flips — this is a property of
  KIVI + greedy sampling, not an integration bug. PPL captures the meaningful
  next-token-prediction quality, which moves only ~2.8%.
- Tokens/sec: **1.55× over vanilla HF** (521.7 vs 335.8). The kernel-level
  Phase 2c INT4 attention was 1.29× over our v3 fp16; the integrated speedup
  is bigger because the forward replacement *also* skips `repeat_kv` and
  the SDPA dispatch overhead, on top of the int4 kernel win itself.
  - **⚠ Phase 7/8 — the 1.55× is `repeat_kv` removal, not the INT4 kernel "on
    top of" it.** Plain fp16 with `enable_gqa=True` (no custom CUDA) already
    scores ~533 tok/s; 4b lands *below* that at 521.7 (0.97× vs fair). The 1.29×
    is CUDA-vs-our-v3, and v3 itself loses to fair SDPA — so the INT4 kernel adds
    no e2e speed on top of the GQA fix; it is a *memory* result. Phase 8's v6 is
    the attention kernel that actually beats fair SDPA (docs/06).
- Peak VRAM: essentially unchanged (-0.09 GB). The INT4 cache really does
  store ~0.27× the bytes of an fp16 cache (Phase 2 proved this and the
  CUDA `quantize_*_int4` kernels here produce the same layout). But peak
  is dominated by prefill, where `cache.update()` materializes the full
  fp16 K/V via `_materialize_fp16` for the SDPA path. After prefill ends,
  steady-state decode memory is mostly the packed storage (so a true
  long-running-decode workload would show the KV memory drop). Avoiding
  the prefill spike would require an INT4 prefill attention kernel —
  scope beyond Phase 4.

**4c notes** (W4A16 weight integration — memory win, batch-sensitive latency):

- Patch (`integration/w4a16_patch.py`): `QuantizedLinear` stores packed INT4
  weights (`[K/8, N] int32`) + per-channel groupwise scales (`[n_groups, N]
  fp16`, group=128) — the Phase 3 scheme. `patch_model_w4a16(model)` walks
  the 32 decoder layers and replaces all 7 projections per layer
  (q/k/v/o_proj on attention, up/gate/down_proj on MLP) — 224 Linears total.
  `embed_tokens` and `lm_head` stay fp16 (no decode dominance, and lm_head
  may share weights with embed).
- `QuantizedLinear.forward` dispatches:
  - M < 256 (decode and small batched-decode): `llmik_cuda.w4a16_gemm`,
    which routes to Phase 3c's decode-optimized kernel at M=1 and to the
    Phase 3b naive kernel at M > 1.
  - M ≥ 256 (prefill): unpack INT4 → multiply by scale → fp16 weight,
    then `torch.matmul` (cuBLAS). The naive kernel's M-cost crosses cuBLAS
    around M ~ 256 on this GPU.
- **Memory win, large and immediate.** After `patch_model_w4a16`:
  weight VRAM goes from **16.06 GB → 5.70 GB** (10.36 GB saved on weights;
  the residual ~2 GB is embed + lm_head, still fp16). Peak during a generate
  call (the row above): **9.05 GB vs vanilla 18.50 GB — 51% reduction.**
  The headline memory result of Phase 4.
- **Accuracy cost compounds with 4b's KIVI noise.** MMLU 68.32 → 62.40
  (-5.92 pp); HellaSwag 79.51 → 77.51 (-2.00 pp); ARC-C 60.84 → 55.72
  (-5.12 pp); PPL 7.055 → 8.087 (+1.03); greedy_match 0.27. The Phase 3
  kernel-level rel-err (42% on synthetic gaussians) softened to a few-percent
  PPL hit on real activations, as predicted; the MMLU drop is in the
  range modern PTQ papers report for similar bit-widths without recovery
  techniques (GPTQ / AWQ both narrow this gap, but neither is implemented
  here).
- **Tokens/sec at batch=16 (the locked workload): 40.9 vs vanilla 335.8 —
  0.12×.** This is a regression. **Root cause:** the Phase 3 decode-optimized
  kernel (`w4a16_gemm_decode_kernel`) is M=1-only; the launcher falls back
  to the Phase 3b naive kernel for M > 1, and the naive kernel does not
  exploit batch parallelism (its M-cost grows linearly per Phase 3 notes).
  Phase 3 RESULTS explicitly flagged this: *"batched-decode M > 1 is
  integration concern (Phase 4)"* — and 4c is where it surfaces.
- **Tokens/sec at batch=1 (per-request decode, where the kernel's design
  assumption holds): 56.9 vs vanilla 48.9 — 1.16×.** Matches the shape of
  Phase 3c's kernel-level wins. The W4A16 win is real *for the design point
  the kernel was built for*.
- **What this teaches.** A decode-shape kernel optimized for M=1 doesn't
  automatically extend to batched-decode serving. Production W4A16 kernels
  (Marlin, AWQ's GEMM-V2, GPTQMarlin in vLLM) all have M-aware fast paths
  for `M ∈ [1, ~128]` — that's what gives them headline tok/s wins at the
  batched-serving point. Building an M-aware batched-decode W4A16 kernel
  is a clean Phase 5 follow-up; the Phase 4c integration proves the rest
  of the plumbing (quant offline, packed loading, forward dispatch, accuracy
  characterization) is correct and ready for that next step.

**Phase 5/6 update** (batched-decode kernel evolution).
The 4c row above shows the current dispatch (Phase 6 v3 kernel for
M ∈ [2, 16], Phase 3c v1 for M=1, Phase 3b v0 for M > 16). History:

| Phase 4c at batch=16 | tok/s | Note |
|---|---:|---|
| M=1 kernel only (commit `8882880`) | 40.9 | M=16 fell back to v0 naive — 16× weight bandwidth amplification |
| + Phase 5 v2 batched-decode kernel | 199.9 | 4.9× recovery; scalar fp32 FMA inner loop |
| + **Phase 6 v3 tensor-core kernel** | **198.7** | Kernel-level 1.3-1.4× over v2, but e2e unchanged — host-side Python/dispatch overhead has moved into the gap |

**Kernel-level vs cuBLAS at M=16** (Phase 6 v3, the on-dispatch kernel):

| Shape | cuBLAS fp16 | Phase 6 v3 | speedup |
|---|---:|---:|---:|
| QKV/O (4096×4096) | 56.8 µs | 152.5 µs | 0.37× |
| MLP up/gate (4096×14336) | 135.8 µs | 167.2 µs | **0.81×** (closest to parity) |
| MLP down (14336×4096) | 179.9 µs | 513.8 µs | 0.35× |

Correctness: max abs err 0.25, mean rel err ~0.0001 (better than v2's
0.001 — fp16 MMA accumulates cleanly in fp32). 51% peak VRAM win and
all accuracy metrics are unchanged.

**Why Phase 6 helps the kernel but not the e2e:**
- Microbench: v3 single-call 153 µs, 7-call back-to-back (one layer's
  worth) 173 µs/call — only 13% inter-call overhead, so launch overhead
  isn't saturating us at the kernel level.
- E2E: ~224 Linear calls per decode step. The kernel went from
  ~64 ms/step (v2) → ~47 ms/step (v3), so 17 ms freed. But we measure
  no e2e speedup → 17 ms of *something else* moved into that gap.
- Best guess: per-`nn.Module.__call__` Python dispatch overhead (~5-30 µs
  × 224 calls × 512 steps ≈ 5-30 seconds of host work across the run).
  At Phase 5's kernel time the GPU was the bottleneck; at Phase 6's the
  host is. **CUDA graphs / `torch.compile`** is the natural next step
  to hide the host overhead — full discussion in
  `docs/04-end-to-end-integration-journey.md`.
  - **⚠ Phase 7 — this host-dispatch mechanism is an untested hypothesis, not
    a measured fact.** A direct check found the vanilla host stall to be **zero**,
    which does not support the 5-30 s Python-dispatch story; the true cause of the
    kernel-vs-e2e gap is still open. Treat the "best guess" as unconfirmed until
    profiled. See docs/05-baseline-correction-journey.md.
