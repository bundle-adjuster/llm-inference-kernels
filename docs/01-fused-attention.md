# Track 1 — Fused Attention

> Write this doc's "Design" and "Optimization roadmap" sections fully before
> writing any kernel code. Fill "Findings" as you go.

## Problem

Attention computes, per head:

```
S = (Q Kᵀ) / sqrt(d)        # [Lq, Lk]
P = softmax(S + mask)        # [Lq, Lk]
O = P V                      # [Lq, d]
```

A naive implementation materialises `S` and `P` in HBM — `O(Lq·Lk)` memory and,
worse, `O(Lq·Lk)` HBM read+write traffic. Attention is **memory-bound**, so that
traffic *is* the runtime.

**FlashAttention** never materialises `S`/`P`: it tiles over blocks of K/V, keeps
a running softmax (max `m` and sum `l`), and rescales a running output
accumulator as each block arrives. HBM traffic drops to streaming Q, K, V, O
once each.

## Two regimes — and why decode is primary

| Regime | Q shape | Character | Kernel |
|--------|---------|-----------|--------|
| **Prefill** | `[Lprompt, d]` | compute-bound for long prompts | FlashAttention-2 forward |
| **Decode** | `[1, d]` | memory-bound: read whole KV cache, tiny FLOPs | FlashDecoding (split-K) |

Chat serving spends most wall-clock in **decode** (one token at a time, long
generations). So the **decode kernel is the primary deliverable**; the prefill
FA-2 forward kernel is a stretch sub-step.

The decode challenge is **occupancy**: work is `batch × n_heads` thread blocks.
For small batch that underfills the GPU. **FlashDecoding** fixes this by
splitting the KV sequence into chunks (split-K): many blocks cooperate on one
head, then a tiny combine step merges their partial softmax states.

## Online softmax (the core trick)

Maintain, while streaming KV blocks, running max `m` and running sum `l`.
For a new block with local max `m_blk`:

```
m_new = max(m, m_blk)
l      = l * exp(m - m_new) + sum(exp(S_blk - m_new))
O_acc  = O_acc * exp(m - m_new) + exp(S_blk - m_new) @ V_blk
m      = m_new
```

Final `O = O_acc / l`. Numerically stable, single pass, no `S` in HBM.
(Milakov & Gimelshein 2018; this is the heart of FlashAttention.)

## Design — decode kernel v0 (naive, must be correct first)

One thread block per `(batch, head)`; a natural block size is `head_dim`
(128) threads. Three phases, with the score vector held in dynamic shared
memory — no online softmax yet (that is optimization 1 below).

1. **Scores.** For each cached key `j`, the block computes
   `s[j] = scale · dot(q, k[b, kv(h), j, :])` — a `head_dim`-wide reduction
   across the block — and stores `s[j]` in shared memory.
2. **Softmax.** Block-reduce `max` over `s`, exponentiate `s[j] − max`,
   block-reduce the sum.
3. **Output.** `o[d] = (1/sum) · Σ_j exp(s[j] − max) · v[b, kv(h), j, d]`.

Details: GQA maps a query head to its kv head via
`kv(h) = h / (n_heads / n_kv_heads)` (`h / 4` for Llama 3 8B). Accumulate in
fp32; inputs and outputs are fp16. The score buffer costs `seqlen_kv · 4`
bytes — beyond the 48 KB static limit, opt in with
`cudaFuncAttributeMaxDynamicSharedMemorySize` (Ada allows ~99 KB).

v0 is intentionally slow — it is the **CUDA-vs-CUDA baseline** the roadmap
below improves, one measured step at a time.

## Optimization roadmap (one commit + one RESULTS.md row each)

1. **Online softmax, single pass** — removes the second KV read.
2. **Warp-level reductions** — `__shfl_down_sync` for dot-products and for
   `m`/`l`; kill shared-memory reduction traffic.
3. **Vectorized loads** — `float4` / 128-bit loads; ensure coalesced, aligned
   KV access. Expect a large jump (this kernel is bandwidth-bound).
4. **Split-K over KV (FlashDecoding)** — multiple blocks per head + combine
   kernel. The decisive win at low batch; target near-peak occupancy.
5. **`cp.async` double-buffering** — overlap KV tile loads with compute
   (Ampere+); hide HBM latency.
6. **(stretch) Tensor Core MMA** — note: `M=1` decode is awkward for MMA
   (wants `M ≥ 8/16`); padding wastes work. Document the tradeoff; CUDA cores
   may stay competitive for decode.
7. **(stretch) Prefill FA-2 forward** — tiled, parallel over Q blocks, warp
   specialization; this is where Tensor Cores clearly win.

## Baselines (CUDA vs Python / SOTA)

- `reference/attention_ref.py` — naive eager attention (correctness oracle).
- `torch.nn.functional.scaled_dot_product_attention` — dispatches to
  FlashAttention / cuDNN.
- `flash_attn` package — FlashAttention-2 and its decode kernel.
- vLLM paged attention kernel — the production decode bar.

## Metrics

- Latency (µs) per decode step.
- **Achieved HBM bandwidth as % of peak** — the headline number for a
  memory-bound kernel; FlashDecoding reaches 80–90%+.
- Model-level tokens/sec when integrated (Phase 4).
- From `ncu`: `dram__throughput.avg.pct_of_peak`, achieved occupancy,
  warp-stall reasons.

## Success criteria

- Threshold: correct; ≥3× over naive PyTorch eager.
- Target: within 20% of `flash_attn` decode on achieved bandwidth.
- Stretch: within 10%; working prefill FA-2 forward.

## References

- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023)
- Dao et al., *Flash-Decoding for long-context inference* (2023, blog)
- Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018)
- vLLM, *PagedAttention* (Kwon et al. 2023)

## Findings

The full narrative — theory, design, measured result, and lesson learned
for each step (including the failed-experiment detours that taught us the
most) — lives in [`01-fused-attention-journey.md`](01-fused-attention-journey.md).
The per-step latency / bandwidth table is in
[`results/RESULTS.md`](results/RESULTS.md).

Current state on `main`: **v3, single-warp block + vectorized 64-bit KV
loads, 0.713 ms / 189 GB/s at the reference workload — 2.34× over the v0
baseline and 1.91× faster than PyTorch SDPA.** Both v4 (FlashDecoding
split-K) and v5 (`cp.async` double-buffer) were explored and reverted —
each regressed by ~50–90 µs across the batch sweep, for different reasons
(bandwidth-bound at the per-SM ceiling for v4; per-iter shmem hop cost
for v5).

Key lessons from this phase:

1. **SIMT-parallel ALU is essentially free** — see the v1 detour where
   moving redundant scalar compute to a single lane *worsened* perf by
   ~190 µs.
2. **`__syncthreads()` is a load barrier the compiler respects** — V loads
   must be hoisted above the sync manually (the v1 prefetch fix).
3. **Removing a shmem broadcast hop can dwarf removing the sync itself** —
   v2's structural change beat the sync-removal prediction by 4×.
4. **Occupancy is a means, not an end** — v3 traded 4 warps/block for 1
   warp/block and won 1.50× because the lost warps were idle at barriers.
5. **Don't add parallelism where the bottleneck isn't compute** — v4's
   split-K added SMs but couldn't unlock more bandwidth from the same pie.
6. **`cp.async` needs heavy per-tile compute to amortise its shmem hop** —
   v5 paid an extra shmem write + read per iter, and nvcc was already
   doing the implicit pipelining anyway, so the explicit version lost.

Still open: headline comparison against `flash_attn`'s decode kernel
(SDPA dispatches to FA/cuDNN but a direct apples-to-apples run is pending),
and `ncu` profile with locked clocks for the "Cause" column in RESULTS.md.
Phase 1's optimization roadmap (v0–v5) is exhausted; the stretch goals —
tensor-core MMA path and prefill FA-2 forward — remain.
