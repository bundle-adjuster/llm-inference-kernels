# Phase 9 Journey — W4A16 End-to-End, Measured Against vLLM

> Companion to [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md)
> (the v6 attention fix) and [`results/RESULTS.md`](results/RESULTS.md).
>
> Phase 8 made the attention kernel competitive with SDPA. Phase 9 asks the
> question the whole repo was built for: **with competitive kernels, can the
> hand-written stack reach vLLM's end-to-end throughput?** The answer is yes on
> throughput — **but only via 4-bit weights, and with caveats that matter.** This
> doc states the win and the caveats with equal weight, because a favorable-metric
> version of this result is exactly the mistake Phase 7 was about.

## The honest scoreboard

Llama 3.1 8B, `batch=16, prompt=512, generate=512`, greedy, RTX 4090.
`benchmarks/bench_w4a16_e2e.py`.

| stack | full e2e tok/s | vs vLLM-fp16 | precision |
|---|---|---|---|
| vanilla HF (fp16) | 343 | 0.49× | fp16 |
| **fair fp16** (v6 attn + `enable_gqa`, no cat) | ~600 | ~0.85× | fp16 |
| vLLM 0.6.6 | **703** | 1.00× | fp16 |
| **W4A16 + v6 + cat-free** (this repo) | **756** | **1.08×** | **int4 weights** |
| — decode-only (same stack) | **843** | **1.20×** | int4 weights |

Two claims, kept separate on purpose:

1. **Kernel quality — matched/beat vLLM, like-for-like.** v6 attention is at
   parity with SDPA (Phase 8); the W4A16 GEMM beats fp16 cuBLAS at M=16 by
   **1.66–2.0×** (below). These are apples-to-apples kernel comparisons and are
   the *solid* result of this phase.
2. **End-to-end throughput — 756 tok/s, above vLLM-fp16's 703 — but 4-bit vs
   16-bit.** This is the repo's thesis realized (weight quantization breaks the
   fp16 weight roofline), *not* a like-for-like win. **W4A16 is lossy** (Phase 4c:
   MMLU 62.4% vs fp16 68.3%); vLLM-fp16 is lossless. A fair 4-bit-vs-4-bit
   comparison is against vLLM-AWQ, which we did **not** measure. And at equal
   precision (fp16 vs fp16) we are still **~0.85×**, short of vLLM.

So: **the kernels are vLLM-class; the e2e "win" is a throughput/accuracy trade,
honestly labeled.**

## How the GEMM got there — split-K, then one-read dequant

The Phase 6 tensor-core W4A16 kernel launched only `ceil(N/64)` blocks — 64 for
the N=4096 Llama shapes, ~0.5 blocks/SM — and streamed weights at **~60 GB/s**.
It *lost* to cuBLAS at M=16, which is why Phase 4c's W4A16 e2e regressed to 199.9
tok/s. The same occupancy wall as v3 attention, the same fix:

1. **Split-K over K** (blockIdx.y): the grid grows by `n_splits×` and fills the
   SMs; each block atomicAdds its fp32 partial, a convert pass casts to fp16.
   60 → ~280 GB/s.
2. **Unpack each packed word once.** The dequant read a full `uint32` once per
   nibble — re-reading every word 8×, leaving the kernel L2/issue-bound. Read it
   once, unpack all 8. 280 → **293–386 GB/s**, and now:

| shape (M=16) | fp16 cuBLAS | W4A16 (ours) | speedup |
|---|---|---|---|
| attn 4096×4096 | 57.3 µs | 28.7 µs | **2.00×** |
| mlp-up 4096×14336 | 134.2 µs | 80.9 µs | **1.66×** |
| mlp-down 14336×4096 | 139.1 µs | 76.0 µs | **1.83×** |

Still only ~35% of peak HBM — the honest headroom (dequant-into-registers,
double-buffering, higher occupancy remain the open Marlin-style work).

## Where the decode step actually goes — and the hypothesis it kills

A profiler pass on the W4A16 decode step (the measurement Phase 6/7 kept
deferring):

- **GPU-busy 19.6 ms of a 19.7 ms step — host stall 0.9 ms (4%).** The step is
  GPU-bound. This **refutes the repo's own Phase 6/7 hypothesis** that ~5–30 ms
  of Python `nn.Module.__call__` dispatch overhead was hiding the W4A16 e2e win.
  It was never host overhead; CUDA graphs would buy ~4%.
- Breakdown: W4A16 GEMMs 65%, the `DynamicCache` `torch.cat` 10%, lm_head 7%, v6
  attention 6%, unfused RMSNorm/RoPE/SiLU/residual ~6%. The remaining gap to
  vLLM is **kernel fusion**, measured, not guessed.

## Killing the cat — the length-aware kernel earns its keep

The `cat` was 10% of the step. A `StaticCache` removes it, but Phase 7 showed a
static cache drops SDPA into its math backend (3.6× slower) — the mask
disqualifies flash. The lever is an attention kernel that takes a **live length**
and reads a contiguous prefix in place. v6 gained exactly that (`seqlen` arg,
`kv_buf_len` stride). `PreallocCache` writes each token in place and hands v6 the
full buffer; v6 attends over `[0, live_len)`.

- **Greedy output bit-identical** to the `DynamicCache` path (100% token match).
- Decode step 22.8 → **19.7 ms**; decode-only 703 → **843 tok/s**.

This is the Phase 7 negative result turned into a win: the thing SDPA couldn't
use, our kernel could.

## Prefill — one fused pass instead of six

W4A16 prefill dequantized weights on the fly in PyTorch (~6 passes + allocation
per matrix): 1.98 s vs fp16's 1.17 s. `w4a16_dequantize` does it in one CUDA pass
— **38× faster** (172 µs vs 6.5 ms on mlp-up), bit-identical. Prefill 1.98 →
**1.09 s**, no longer a regression. This is what lifts *full* e2e (not just
decode) above vLLM-fp16.

## What survives, and what's still open

- **Solid:** the kernels are vLLM-class — v6 == SDPA, W4A16 GEMM > cuBLAS,
  cat-free decode, fused prefill. All correctness-gated, all in `main`.
- **True but caveated:** W4A16 e2e (756) > vLLM-fp16 (703) — 4-bit vs 16-bit, a
  throughput/accuracy trade, not a like-for-like or a quality match.
- **Open:** (1) fp16-vs-fp16 parity needs the unfused elementwise fused
  (~6% of the step); (2) the W4A16 GEMM is at ~35% of peak — Marlin-style
  register dequant is the path to the thesis's projected ~2× over vLLM; (3) a
  fair 4-bit comparison vs vLLM-AWQ; (4) `ncu` with locked clocks.

## Reproduction

```bash
scripts/lock_clocks.sh
python benchmarks/bench_w4a16.py            # W4A16 GEMM vs cuBLAS (M sweep)
python benchmarks/bench_w4a16_e2e.py        # full stack vs vLLM-fp16, with caveats
```
