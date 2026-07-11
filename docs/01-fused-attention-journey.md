# Phase 1 Journey — Fused Decode Attention

> This document complements [`01-fused-attention.md`](01-fused-attention.md)
> (the forward-looking design) and [`results/RESULTS.md`](results/RESULTS.md)
> (the numbers). It captures the **journey**: what each step's theory was,
> what we built, what we measured, and what we learned — especially when
> the result surprised us.
>
> Every numbered step corresponds to a git commit; hashes are inline so you
> can `git show <hash>` for the diff.
>
> ---
>
> **⚠ Read this first (added in Phase 7).** The baseline this entire document
> races against — "PyTorch SDPA, 1.36 ms" — is SDPA handed a **4×-expanded GQA
> key/value cache** (`reference/attention_ref.py:94`, `_expand_gqa`). SDPA's
> native GQA path does the same work in **157.7 µs**. Against it, v3 is **4.55×
> slower**, not 1.91× faster.
>
> Worse, the v4 diagnosis below ("bandwidth was the ceiling") is false: flash's
> `flash_fwd_splitkv_kernel` reaches **81% of peak HBM** on this exact workload
> where v3 reaches 18%. **v3 is occupancy-bound** — the single-warp block it
> adopts in the v3 step caps per-SM occupancy at 33%. Split-K was the right
> idea, applied to a block geometry that had already given the parallelism away.
>
> The CUDA-vs-CUDA story below (v0 → v3, and *why* each step moved) is sound and
> worth reading. Only the "vs SDPA" column and the v4 conclusion are wrong.
> Full analysis: [`05-baseline-correction-journey.md`](05-baseline-correction-journey.md).
>
> **✅ Phase 8 fixed it.** The v6 kernel (`kernels/attention/fused_attention_splitk.cu`)
> is FlashDecoding split-K on **multi-warp blocks** (4 warps/block) with a 4-deep
> unrolled load loop — exactly the direction v3 closed off. On this reference
> workload v6 is **155.6 µs vs GQA-native SDPA's 157.3 µs (1.01×, beats it)** at
> **~82% of peak HBM** (v3 was 18%), and **4.59× over v3**. The binding
> `decode_attention` now dispatches to v6; the old kernel is preserved as
> `decode_attention_v3`. See
> [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

## Setting

- **GPU**: RTX 4090 (sm_89), 1008 GB/s peak HBM, 72 MB L2.
- **Reference workload**: Llama 3.1 8B head config — `n_heads=32`,
  `n_kv_heads=8`, `head_dim=128`. Microbench at `batch=8, seqlen_kv=4096`,
  fp16.
- **What we measure**: median latency (CUDA events, 25 warmup + 100 timed
  iters); `max |abs diff|` vs the fp32 reference; achieved "KV bandwidth" =
  `(|K| + |V|) / median_latency` (logical, not raw HBM).
- **What we're racing**: PyTorch SDPA, which dispatches to
  FlashAttention/cuDNN. On our reference workload it lands at **1.36 ms**.
  We also report PyTorch eager (3.77 ms) for context.

The decode attention task, per `(batch, head)`:
```
S[j] = scale · dot(q, K[b, kv(h), j, :])    for j in [0, seqlen_kv)
P    = softmax(S)
o[d] = Σ_j P[j] · V[b, kv(h), j, d]
```
GQA: `kv(h) = h / (n_heads / n_kv_heads)` (`h / 4` for Llama 3 8B).
fp16 in/out, fp32 accumulation.

---

## v0 — naive two-pass softmax  ·  commit `46930c2`

### Theory

Textbook three-phase decode: (1) compute the full score vector `s[j]` and
materialise it in shared memory, (2) block-reduce `max(s)` and `Σ exp(s−max)`
for the softmax, (3) weighted sum `Σ_j P[j] · V[j, d]`. The score buffer is
the only stateful piece outside registers; it costs `seqlen_kv · 4 B` of
dynamic shmem.

### Design choices

- **Block geometry**: `grid(batch, n_heads), block(head_dim)` = 128 threads.
  Each thread owns one Q lane and one output lane.
- **Cross-warp reduction**: per-warp shfl, then lane 0 of each warp writes
  `reduce_smem[warp_id]`, sync, warp 0 reduces, lane 0 writes `s_smem[j]`,
  sync.
- **GQA mapping**: `kv_head_idx = head_idx / (n_heads / n_kv_heads)`.
- **fp32 accumulators**, fp16 inputs/outputs (the design doc is explicit:
  half-accumulation would lose accuracy).

### Result

**1.669 ms / 80 GB/s** at the reference workload. Max `|abs diff|` = 6.1e-5
(well under the `rtol/atol=2e-2` gate). **2.26× faster than PyTorch eager
(3.77 ms); 23% behind SDPA (1.36 ms).**

### What we learned

- This is the CUDA-vs-CUDA baseline the roadmap is meant to beat.
- The shmem score buffer caps `seqlen_kv` at ~12k positions before we'd
  need the `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in.
- Phase 1 (scores) reads K with two `__syncthreads()` per `j`. Phase 3
  (output) reads V in a sync-free loop — natural compiler pipelining for V.
  This asymmetry will matter in v1.
- Pre-existing infrastructure bug found en route: `setup.py` used a
  relative `include_dirs=["kernels"]` that ninja never resolved (ninja runs
  from `build/temp/...`). Fixed to absolute path; this is what unblocked
  v0 from building at all.

---

## v1 — online (streaming) softmax  ·  commits `46ae1ea`, `ad9c57f`

### Theory (Milakov & Gimelshein 2018)

Replace the two-pass softmax with a running recurrence over `j`:
```
m_new = max(m, s_j)
alpha = exp(m − m_new)                  // rescale prior accumulators
p_j   = exp(s_j − m_new)
O_acc = O_acc · alpha + p_j · v[j, :]
l     = l     · alpha + p_j
m     = m_new
```
At the end: `out = O_acc / l`. **Single pass over KV, no score buffer in
shmem, unbounded `seqlen_kv`.** Theoretically it cannot be slower than v0 —
it does strictly less work.

### First port (commit `46ae1ea`) — **regression**

Every thread computes its own `(m, l, alpha, p_j)` from the broadcast `s_j`.
Each thread maintains its own running state in registers. Identical state
evolves across threads because they all see the same `s_j`.

**Result: 2.078 ms / 65 GB/s — 0.80× of v0.**

### The wrong hypothesis (a useful detour)

Initial diagnosis: "the recurrence is per-thread, so 128 threads × 4096 `j`s ×
2 `__expf` calls per `j` = 256× more `expf` work than v0's phase 2." The
proposed fix moved the recurrence to lane 0 of warp 0 and broadcast
`(alpha, p_j)` via shmem.

**That made it worse: 2.26 ms.**

**Lesson**: under SIMT, parallel-identical ALU work is essentially free.
Those warps were going to wait at the `__syncthreads()` anyway, so doing
the same scalar compute alongside cost nothing in wall-clock. Concentrating
it in one lane introduced a serial-on-lane-0 critical path and an extra
shmem-broadcast hop, with three idle warps. We disproved the hypothesis by
testing it — the failed experiment is more useful than the wrong intuition.

### The right fix — V prefetch (commit `ad9c57f`)

The actual culprit was structural, not arithmetic. v0 phase 3 read V in a
sync-free loop the compiler could deeply pipeline. v1 put the V load
*inside* the per-`j` `__syncthreads()`. **nvcc will not hoist a memory load
across `__syncthreads()` on its own** — so V's latency was fully serialised
behind the sync.

The fix: issue `v_j = V[j, tid]` at the *top* of the iteration, right
alongside `k_j`. The load is non-blocking; the value is consumed only after
both syncs and the softmax update, so its latency overlaps with all of
that work.

### Result (with V prefetch)

**1.637 ms / 82 GB/s — 1.02× over v0.** Max `|abs diff|` = 3.1e-5. Now
consistent with theory: single-pass, no shmem score buffer, sync-overlapped
V latency.

### What we learned

1. **The first hypothesis was wrong; the experiment disproved it cheaply.**
   The "redundant exp" theory predicted a fix that *worsened* perf —
   strong signal we were diagnosing the wrong thing.
2. **nvcc respects `__syncthreads()` as a load barrier.** Hoisting a
   memory load above a sync is your job, not the compiler's.
3. **SIMT-parallel ALU is free.** The same lesson recurs in v2.

---

## v2 — single-sync block reduce  ·  commit `db6ab0b`

### Theory

v1 still has two `__syncthreads()` per `j`: one after each warp writes its
partial to `reduce_smem`, one after warp 0 writes the broadcast `s_bcast`.
Both barriers cost time. Can we drop to one?

Two ideas combine:
1. **All warps redundantly do the final cross-warp reduce.** After the
   first sync, every thread reads `reduce_smem[0..n_warps]` (with
   `lane_id >= n_warps` masked to 0), runs `warp_reduce_sum`, and arrives
   at the block-wide `s_j` via shuffle. No shmem broadcast slot, no
   second sync. The redundant compute across warps is essentially free
   (SIMT lesson from v1).
2. **Double-buffer `reduce_smem` on `j & 1`.** With a single buffer, iter
   j+1's write would race iter j's still-in-flight reads in slower warps.
   With two slots, iter j+1 writes the *other* slot. The hazard between
   iter j's read and iter j+2's write (same slot) is gated by iter j+1's
   sync — j's reads must finish before j+1's sync, before j+2 even starts.

### Result

**1.069 ms / 126 GB/s — 1.53× over v1-prefetch, 1.56× over v0.** Now
**1.27× faster than PyTorch SDPA.** Max `|abs diff|` = 3.1e-5. Register
count 29 → 28; shmem 132 B → 256 B (the second buffer slot).

> **⚠ Phase 7 / ✅ Phase 8:** the "1.27× faster than PyTorch SDPA" here is
> against the 4×-expanded-GQA baseline; GQA-native SDPA is 157.7 µs, so v2
> actually loses. The CUDA-vs-CUDA ratios (1.53× over v1, 1.56× over v0) stand.
> The kernel that finally beats fair SDPA is v6 split-K —
> [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

### What surprised us

The eliminated sync alone predicted ~150 µs of savings. We measured ~570 µs.
The extra came from removing the `s_bcast` shmem hop: in v1, `s_j` went
*shfl tree → lane 0 writes shmem → sync → all threads read shmem*. In v2,
`s_j` flows directly from the shfl tree into each thread's softmax FMA. No
shmem round-trip in the critical path.

### Lesson

Removing an entire shmem hop can dwarf the visible "sync removal." If you
can structure the data flow so that a result lives in a register all the
way to its consumer, do it.

---

## v3 — vectorized 64-bit KV loads  ·  commit `ccdb6df`

### Theory

In v2, each warp-wide K load is 64 bytes (32 lanes × 2 bytes), and the
128-thread block issues 4 of these in parallel. That's 4 load instructions
per `j`. Vectorizing each thread's load to 8 bytes (`LDG.E.64` via `uint2`)
collapses one warp's load to a single 256-byte instruction. Fewer
instructions for the same bytes → higher load throughput, especially on
the L1 path.

But head_dim=128 with a 128-thread block leaves no room for per-thread
vec=4 (each thread already covers 1 d-lane). To vectorize we must change
the thread/lane mapping.

### Design choices

- **Block shrinks 128 → 32 (single warp).** Each thread now owns
  `head_dim / 32 = 4` d-lanes — the natural `VEC` width.
- **`uint2` loads**, unpacked via `__half22float2` to float4.
  ```cuda
  float4 load_half4_as_float4(const half* ptr) {
      uint2 raw = *reinterpret_cast<const uint2*>(ptr);
      half2 lo, hi;
      *reinterpret_cast<uint*>(&lo) = raw.x;
      *reinterpret_cast<uint*>(&hi) = raw.y;
      float2 f_lo = __half22float2(lo);
      float2 f_hi = __half22float2(hi);
      return make_float4(f_lo.x, f_lo.y, f_hi.x, f_hi.y);
  }
  ```
  Vec store is the inverse (`STG.E.64`).
- **Single-warp block means no `__syncthreads()` at all.** Shmem usage
  drops to 0 B. `warp_reduce_sum` alone does the dot-product reduce.
- **V prefetch still applies**: issue `v_v` at the top of the iter.

### The occupancy bet

Per-SM occupancy collapses from full (12 blocks × 4 warps = 48 warps) to
~16 warps (16 blocks × 1 warp = 33%). The bet: vec throughput plus sync
removal beats the lost latency-hiding warps. We measured before merging.

### Result

**0.713 ms / 189 GB/s — 1.50× over v2, 2.34× over v0, 1.91× over PyTorch
SDPA.** Max `|abs diff|` = 3.1e-5. Register count 28 → 33; shmem 256 B → 0 B.

> **⚠ Phase 7 / ✅ Phase 8:** the "1.91× over PyTorch SDPA" is against the
> expanded-GQA baseline. Against GQA-native SDPA (157.3 µs) v3 is **4.55×
> slower** (0.22×) — the single-warp block is **occupancy-bound** (~2 of 128
> SMs, 18% of peak HBM), not bandwidth-bound. The 1.50×/2.34× CUDA-vs-CUDA
> ratios remain valid. Phase 8's v6 restores multi-warp blocks + split-K and
> hits **155.6 µs (1.01×, beats fair SDPA), 4.59× over v3, ~82% of peak HBM** —
> [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

### What we learned

- The lost latency-hiding warps weren't doing useful work for us — they
  were sitting at `__syncthreads()` barriers most of the time. **Occupancy
  is a means, not an end.**
- One vectorized 64-bit load per warp per `j` is a sweet spot for our
  layout. Going further (vec=8 = 128-bit, `LDG.E.128`) requires processing
  2 `j` positions per warp-iter, which bumps into the same split-K-style
  state management as v4 — not strictly forbidden, but a bigger restructure.

---

## v4 — split-K (FlashDecoding) — **explored, reverted**  ·  commit `f904aae` (revert `e39c97f`)

### Theory

v3 launches `batch × n_heads = 256` blocks. With ~16 blocks/SM that's
about 16 SMs busy out of 128. For small batch the canonical fix is
**FlashDecoding**: split each `(batch, head)` across `K_SPLIT` chunks of
the KV sequence; the grid becomes `batch × n_heads × K_SPLIT = 2048`
blocks; a tiny combine kernel merges `K_SPLIT` partial softmaxes.

The combine formula (online softmax across splits):
```
m_final = max_s m_s
l_final = Σ_s l_s · exp(m_s − m_final)
o_final = (Σ_s o_s · exp(m_s − m_final)) / l_final
```

### Design choices

- Two kernels:
  - **Stage 1**: grid `(batch, n_heads, K_SPLIT)`, block 32. Same v3 body
    but bounded to `j ∈ [s · chunk, (s+1) · chunk)`. Writes unnormalized
    `o_acc[head_dim]` plus `(m, l)` to scratch.
  - **Stage 2**: grid `(batch, n_heads)`, block 32. Reads K_SPLIT partials,
    applies the combine formula, writes the final normalized output.
- **K_SPLIT = 8** (constexpr). Tried 2, 4, 8, 16; 8 and 16 tied at best,
  4 was worse than v3 by a wide margin.
- **Scratch**: ~528 KB. Allocated stream-ordered via `cudaMallocAsync`.
  Consolidated 3 separate allocations into 1 — no measurable effect.

### Result — regression at every batch tried

| batch | SDPA   | v3       | v4       |
|------:|-------:|---------:|---------:|
| 1     | 0.135  | 0.467 ms | 0.711 ms |
| 2     | 0.330  | 0.463 ms | 0.712 ms |
| 4     | 0.696  | 0.441 ms | 0.683 ms |
| 8     | 1.362  | 0.715 ms | 0.802 ms |

v4's wall-clock is roughly *flat* at ~0.7 ms across batch sizes — there's
~250 µs of fixed overhead per call that doesn't unlock anything new.

### Diagnosis — why FlashDecoding didn't help here

v3 wasn't actually grid-undersized at our workload. At batch=1, ~11 SMs are
busy under v3, but those SMs were already at the **HBM/L2-throughput
ceiling** for the (small, ~16 MB total) KV set, not under-occupied on
compute. Adding 8× more blocks via split-K spreads the same constrained
bandwidth across more SMs — per-SM throughput drops proportionally, total
stays roughly the same, and the second kernel launch + scratch traffic
add a net ~250 µs of overhead.

The clue: at batch=1, v3 achieves 36 GB/s of effective KV bandwidth and v4
gets *less* — 24 GB/s. More SMs is not more bandwidth when bandwidth is
the shared bottleneck.

What was *not* the issue: scratch allocation overhead. Consolidating 3
`cudaMallocAsync` calls into 1 was a no-op for perf.

### Lesson

**Split-K is a tool for grid-undersized workloads — but "undersized" means
"the GPU has idle compute that more parallelism could exploit."** If the
per-SM bandwidth is already the ceiling, adding more SMs just splits the
same pie. Our `batch × n_heads = 256` workload was bandwidth-bound, not
grid-bound, even at batch=1.

This optimization belongs in the toolkit for workloads where v3 truly
underfills the GPU *and* has bandwidth headroom (e.g. tiny batches on
GPUs with very high SM counts and low per-SM bandwidth pressure).

> **⚠ Phase 7 / ✅ Phase 8 — this lesson was the misdiagnosis.** Our
> `batch × n_heads = 256` workload was **not** bandwidth-bound: a single-warp
> block fills ~2 of the 4090's 128 SMs, so it was *occupancy*-bound, with the GPU
> ~96% idle. Fair GQA-native SDPA reaches 81% of peak HBM on the identical bytes
> where v3 reaches 18% — the ceiling blamed here does not exist. Split-K **was**
> the right tool; v4 just bolted it onto the single-warp block that had already
> given the occupancy away. Phase 8's **v6** does split-K on 4-warp blocks and
> **beats fair SDPA** (155.6 µs, 1.01×, 4.59× over v3, ~82% of peak). See
> [`05-baseline-correction-journey.md`](05-baseline-correction-journey.md) and
> [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

### Decision

The v4 source remains in git history at `f904aae` for reference. `e39c97f`
reverts the kernel to v3 so `main` runs the fast path.

---

## v5 — `cp.async` double-buffered KV tiles — **explored, reverted**  ·  commit `78a28ff` (revert follows)

### Theory

v3's V prefetch papers over load latency via the compiler scheduler:
nvcc can hoist *one* load before its consumer, hiding it behind the
warp_reduce_sum, but it can't pipeline N tiles deep because each iter
depends on the previous's `(m, l, O_acc)`. **`cp.async`** (Ampere+) is the
hardware feature that exposes deeper pipelining: it issues a load
asynchronously to shared memory, lets the thread keep going, and
`__pipeline_wait_prior(N)` blocks just enough — only when the data is
actually needed.

The standard pattern is **double buffering**: while iter j processes its
tile (read from slot `j & 1`), iter j+1's load is already in flight to
the *other* slot.

### Design choices

- Two-slot shmem buffer: `extern __shared__ half kv_smem[]` of size
  `NUM_STAGES × head_dim × 2 (K + V) × 2 B = 1 KB` total.
- Per-thread cp.async of 8 bytes (one `LDG.E.64` equivalent): each thread
  copies only its own 4-half slice of the tile, and only reads its own
  slice back. No cross-thread shmem traffic, so no `__syncwarp()` between
  wait and consume.
- Pipeline: prime tile 0 → slot 0; in the loop, prefetch tile j+1, then
  `__pipeline_wait_prior(1)`, read tile j from `slot (j & 1)`. Last iter
  has no prefetch and uses `wait_prior(0)`.

### Result — regression at every batch tried

| batch | SDPA  | v3       | v5       |
|------:|------:|---------:|---------:|
| 1     | 0.135 | 0.467 ms | 0.536 ms |
| 2     | 0.328 | 0.463 ms | 0.536 ms |
| 4     | 0.695 | 0.441 ms | 0.506 ms |
| 8     | 1.359 | 0.715 ms | 0.760 ms |

Uniformly ~50 µs slower than v3.

### Diagnosis — why deeper pipelining didn't help

**cp.async writes only to shmem.** Where v3 went `global → register`
directly, v5 goes `global → shmem (cp.async) → register (load)`. That's
one extra shmem write *and* one extra shmem read per iter. At ~10 cycles
combined × 4096 iters = ~40k cycles per block — measurable.

What we expected to win: explicit overlap of the next tile's load with
this tile's compute, beyond what nvcc was already doing in v3 via
ordinary load latency hiding (the inline V prefetch from v1).

What actually happened: nvcc's compiler-scheduled pipelining in v3 was
already capturing most of the available overlap. Making the pipeline
explicit didn't expose new headroom — but the shmem hop was a new fixed
cost. cp.async also bypasses L1 (goes L2-only). Neutral here since K+V
per head (~2 MB) doesn't fit L1 (128 KB) anyway, but it's worth knowing
as a tradeoff.

We did *not* try `NUM_STAGES = 3` or 4. Deeper pipelining hides more load
latency but doesn't reduce the per-iter shmem hop cost — that's
depth-independent. With shmem cost dominating, more depth wouldn't move
the needle.

### Lesson

**`cp.async` is the wrong tool when per-iter compute is small relative to
per-iter shmem access cost.** It shines when the compute per tile is
heavy enough to fully overlap N tile loads (e.g., MMA-heavy prefill or
GEMM), so the shmem hop is amortised. For decode attention with
short per-iter compute, the compiler-scheduled global load + register
pipelining in v3 is already near-optimal for this access pattern.

The other condition for `cp.async` to win: load latency must be the
bottleneck. For us, ~189 GB/s of 1008 peak is only 19% — so latency
isn't fully hidden — but the per-SM bandwidth is the ceiling (we proved
that in v4), and `cp.async` doesn't change per-SM bandwidth.

> **⚠ Phase 7 / ✅ Phase 8:** "the per-SM bandwidth is the ceiling (we proved
> that in v4)" is false — v4 proved no such thing. v3 was **occupancy-bound**
> (18% of peak HBM, ~2 of 128 SMs), not bandwidth-bound; flash reaches 81% on the
> same bytes. `cp.async` didn't help because a single warp per block keeps too
> few loads in flight, not because HBM was saturated — the real fix was **more
> warps and more blocks** (Phase 8's v6 split-K, ~82% of peak). See
> [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

### Decision

v5 source preserved at `78a28ff`; follow-up commit restores v3 on `main`.

---

## Summary

> **⚠ The `vs SDPA` column is against the expanded-GQA baseline and is inflated
> ~8.7×.** GQA-native SDPA is 157.7 µs (812 GB/s); every kernel below loses to
> it. The `vs v0` column is CUDA-vs-CUDA and remains valid.
> **✅ Phase 8:** the v6 split-K kernel (not in this table) lands at 155.6 µs —
> **1.01×, the first to beat fair SDPA** — see
> [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

| Step                        | Latency   | BW (GB/s) | vs v0    | vs SDPA  |
|-----------------------------|----------:|----------:|---------:|---------:|
| **SDPA, GQA-native (real SOTA)** | **0.158 ms** | **812** | **10.6×** | **8.62×** |
| PyTorch eager               | 3.77 ms   |    —      | 0.44×    | 0.36×    |
| PyTorch SDPA (expanded GQA) | 1.36 ms   |    —      | 1.23×    | 1.00×    |
| v0 (two-pass softmax)       | 1.669 ms  |    80     | 1.00×    | 0.82×    |
| v1 naive (regressed)        | 2.078 ms  |    65     | 0.80×    | 0.66×    |
| v1 + V prefetch             | 1.637 ms  |    82     | 1.02×    | 0.83×    |
| v2 (single-sync reduce)     | 1.069 ms  |   126     | 1.56×    | 1.27×    |
| **v3 (vec loads, 1-warp)**  | **0.713 ms** | **189** | **2.34×** | **1.91×** (Phase 7: retired, 0.22× vs fair SDPA; Phase 8 v6 beats it — see docs/06) |
| v4 split-K (reverted)       | 0.802 ms  |   167     | 2.08×    | 1.70×    |
| v5 cp.async (reverted)      | 0.760 ms  |   177     | 2.20×    | 1.79×    |

### Lessons we want to carry forward

1. **SIMT-parallel ALU is essentially free.** If warps would otherwise wait
   at a sync, having them do identical scalar compute alongside isn't
   waste — moving the compute to one lane often introduces a *worse*
   serialization (v1 detour, v2 confirmation).

2. **`__syncthreads()` is a load barrier the compiler respects.** Memory
   loads consumed after a sync must be issued explicitly *before* the sync
   if you want their latency to overlap with the reduction (v1 V prefetch).

3. **Removing a shmem hop can dwarf removing a sync.** The single-sync
   reduce in v2 saved ~150 µs from one fewer `__syncthreads()` but ~420 µs
   from no longer routing `s_j` through shmem.

4. **Occupancy is a means, not an end.** Trading 4 warps/block for 1 can
   win when the lost warps were idle at barriers and the win is in load
   throughput (v3).

5. **Don't add parallelism where the bottleneck isn't compute.** Split-K
   helps when the grid is compute-undersized *and* bandwidth has headroom.
   If per-SM HBM/L2 throughput is the ceiling, more SMs just split the same
   pie (v4).

6. **`cp.async` requires heavy per-tile compute to amortise the shmem
   hop.** It writes only to shmem, so you pay an extra shmem write + read
   per tile. For decode attention with light per-iter compute, the hop
   costs more than the explicit pipelining wins — and nvcc was already
   doing the available overlap implicitly through the compiler schedule
   on direct register loads (v5).

7. **Test the wrong hypothesis cheaply.** v1's "redundant exp" theory
   would have been our headline optimization had we not measured it. The
   measurement disproved it; the disproof taught us the SIMT lesson.

### What's still open

- **Headline number vs `flash_attn`'s own decode kernel.** SDPA dispatches
  to FA/cuDNN; we want to plug `flash_attn` in directly for an apples-to-
  apples comparison. The `flash_attn` row in `RESULTS.md` is still TBD.
- **`ncu` profile.** Every "Cause" cell in `RESULTS.md` says "pending"
  because we haven't run with locked clocks. Profiler-backed causes are
  worth more than reasoned ones.
- **Phase 1 optimization roadmap is exhausted.** v0–v5 are all tried;
  v3 is the kernel on `main`. Phase 1 declared done at **0.713 ms / 189
  GB/s, 1.91× faster than PyTorch SDPA on the reference workload.** The
  stretch goals — tensor-core MMA path, prefill FA-2 forward kernel —
  remain open in the docs/01 roadmap.

  > **⚠ Phase 7 reopens this.** The roadmap was declared exhausted against a
  > handicapped baseline, and against a bandwidth ceiling that does not exist.
  > v3 sits at 18% of peak HBM; flash's split-KV kernel reaches 81% on the same
  > bytes. The unexplored direction is the one v3 closed off: **multi-warp
  > blocks to recover the 33% occupancy, then split-K over the sequence.**
  > That is what FlashDecoding does and what our v4 tried to bolt onto a
  > one-warp block. See
  > [`05-baseline-correction-journey.md`](05-baseline-correction-journey.md).
  >
  > **✅ Phase 8 closed it.** That exact direction — 4 warps/block + split-K with
  > a 4-deep unrolled load loop — became **v6**
  > (`kernels/attention/fused_attention_splitk.cu`). Phase 1 is *not* done at
  > "0.713 ms, 1.91× faster"; the true endpoint is **v6 at 155.6 µs, 1.01× over
  > GQA-native SDPA (the first fair-baseline win), 4.59× over v3, ~82% of peak
  > HBM**, within 2.4e-4 of SDPA. v6 beats fair SDPA on HBM-bound shapes
  > (kv≥2048) and trails on L2-resident shapes (kv≤1024, 0.69–0.82×) — that L2
  > gap is the honest remaining boundary. Full writeup:
  > [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).
