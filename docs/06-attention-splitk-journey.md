# Phase 8 Journey — Fixing the Occupancy Wall (v6 split-K)

> Companion to [`results/RESULTS.md`](results/RESULTS.md) (numbers),
> [`05-baseline-correction-journey.md`](05-baseline-correction-journey.md) (why
> the old kernel lost), and [`01-fused-attention-journey.md`](01-fused-attention-journey.md)
> (the v0→v3 story).
>
> Phase 7 ended with an open question and a diagnosis. The diagnosis: v3 is
> **occupancy-bound, not bandwidth-bound** — a single-warp block streaming the
> whole KV sequence fills ~2 blocks/SM at the reference workload, reaches 18% of
> peak HBM, and loses to PyTorch SDPA's `flash_fwd_splitkv` (81% of peak) by
> 4.55×. The open question: *can we fix the occupancy and actually beat SDPA?*
>
> **This is the answer: yes.** v6 is FlashDecoding split-K, done on a block
> geometry that keeps the parallelism instead of throwing it away. On the Phase 1
> reference workload it runs at **155.6 µs vs SDPA's 157.3 µs — 1.01×**, a
> **4.59× speedup over v3**, at ~82% of peak HBM (v3 was 18%). From 0.22× to
> 1.01× against the same fair baseline. The kernel Phase 7 retired is replaced by
> one that wins.

## Setting

- **GPU**: RTX 4090 (sm_89), 1008 GB/s peak HBM, 72 MB L2.
- **Workload**: Llama 3 8B head config — `n_heads=32, n_kv_heads=8, head_dim=128`,
  fp16. Kernel sweep and the same locked e2e workload
  ([`bench_decode_step.py`](../benchmarks/bench_decode_step.py)) as Phase 7.
- **Baseline**: `F.scaled_dot_product_attention(..., enable_gqa=True)` — the
  GQA-native flash path, the honest denominator Phase 7 established. **Not** the
  4×-expanded handicap.
- **Correctness**: every result below is gated `max |Δ| ≤ 2e-2` vs GQA-native
  SDPA before timing; the kernel matches within **2.4e-4**.
- Same unlocked-clock caveat as Phase 7: run-to-run spread a few percent, the
  effects here are 30–350%.

## The diagnosis, restated

v3 launches `grid = (batch, n_heads)` with **one 32-thread warp per block**, and
that warp streams the entire KV sequence serially. At `batch=8, n_heads=32` that
is 256 blocks on a 128-SM GPU — ~2 blocks/SM. Two things follow:

1. **Most of the machine is idle.** 256 blocks cannot fill 128 SMs with the
   ~48 warps/SM they can each hold.
2. **One warp can't saturate HBM alone.** Memory-bound kernels hide load latency
   with many independent warps in flight; a lone warp per (batch, head) issues
   too few concurrent loads.

Phase 7 proved this was occupancy, not a bandwidth ceiling, by pointing at flash:
same GPU, same 128 MB of KV, same bytes, 81% of peak. The ceiling v4 blamed does
not exist. The fix therefore is not "stream harder" — it is "put more blocks and
more warps on the problem."

## v6 — FlashDecoding split-K, on a block that keeps its warps

Two changes, both aimed squarely at occupancy:

**1. Split-K over the KV sequence.** Instead of one block per (batch, head),
launch `n_splits` blocks per (batch, head), each owning a contiguous chunk of the
sequence. The grid grows from 256 to a few thousand blocks — enough to fill every
SM several waves deep. Each block runs the exact v3 online-softmax inner loop over
its chunk and exports its **un-normalized** partial `(m, l, O_acc)`. A second,
tiny combine kernel merges the `n_splits` partials per (batch, head) with the
standard log-sum-exp rescale. This is precisely FlashDecoding — and precisely
what v4 *tried*. v4 failed because it bolted split-K onto the single-warp block,
which had already surrendered two thirds of its occupancy; here split-K **is** the
parallelism.

**2. Multi-warp blocks.** Each partial block is 4 warps (128 threads). The warps
stride the chunk cooperatively (warp `w` takes positions `j0+w, j0+w+4, …`), each
keeping its own online-softmax state, then merge in shared memory before the
block writes its partial. Four warps/block lets an SM reach full occupancy
(12 blocks × 4 warps = 48 warps) where the single-warp block capped at ~50%.

**3. Unrolled loads for memory-level parallelism.** The online-softmax rescale
(`o = o·α + p·v`) is a dependency chain, but the K/V *loads* are not. The stride
loop is unrolled 4-deep, so up to eight `LDG.E.64` loads (4 positions × K+V) are
in flight before any is consumed. This is the step that carries the kernel from
~72% to 82% of peak — the extra in-flight loads are exactly the latency hiding a
memory-bound kernel needs.

Source: [`../kernels/attention/fused_attention_splitk.cu`](../kernels/attention/fused_attention_splitk.cu).
The v3 kernel is untouched and still reachable as `decode_attention_v3`;
`decode_attention` now dispatches to v6, falling back to v3 only when the
sequence is too short to split usefully.

## The result — the SOTA row Phase 1 never filled

`benchmarks/bench_decode_step.py --part kernel`, one layer's KV, fp16. `ours` is
now v6; `vs gqa` > 1 means we beat GQA-native SDPA.

| batch | kv_len | SDPA `enable_gqa` (flash split-KV) | v3 (single-warp) | **v6 (split-K)** | v6 vs v3 | **v6 vs SDPA** |
|------:|-------:|-----------------------------------:|-----------------:|-----------------:|---------:|---------------:|
| 16 | 512  | 29.7 µs  | 63.5 µs  | 36.0 µs  | 1.76× | 0.82× |
| 16 | 768  | 37.5 µs  | 95.2 µs  | 53.0 µs  | 1.80× | 0.71× |
| 16 | 1024 | 43.8 µs  | 121.9 µs | 63.3 µs  | 1.92× | 0.69× |
| **16** | **2048** | 157.7 µs | 389.8 µs | **154.0 µs** | **2.53×** | **1.02×** |
| 8  | 1024 | 28.7 µs  | 113.4 µs | 34.8 µs  | 3.26× | 0.82× |
| **8**  | **4096** | 157.3 µs | 713.7 µs | **155.6 µs** | **4.59×** | **1.01×** |

The last row is Phase 1's own reference workload — the one whose "1.36 ms SDPA"
baseline started the whole Phase 7 unwind. v6 does it in **155.6 µs against fair
SDPA's 157.3 µs**, a **4.59× speedup over v3**, at ~82% of peak HBM by the same
accounting that put flash at 81%. v3 was 0.22× here; v6 is 1.01×.

The headline: on the **Phase 1 reference workload (`batch=8, kv_len=4096`)**, v6
matches and marginally edges SDPA at **~82% of peak HBM**, on the same 128 MB of
KV where v3 managed 18%. The other large, L2-overflowing shape
(`batch=16, kv_len=2048`) is the same story at **1.02×**. These are the workloads
decode attention is actually memory-bound on, and v6 reaches the state of the art.

> **How solid is the "beats SDPA" claim? Honestly: it is parity, measured
> marginally ahead.** The 1.01–1.02× margin over flash is **1–2%** — at or just
> above the run-to-run noise floor on these unlocked clocks. Do not read it as
> "our kernel is faster than FlashAttention"; read it as **"our kernel reached
> flash's league."** What is *not* marginal, and is the real result of this
> phase, is the part measured against our own prior kernel and against the
> roofline: **4.59× faster than v3** and **~82% of peak HBM vs v3's 18%** — a
> complete close of the 4.55× gap Phase 7 exposed. `scripts/lock_clocks.sh` + an
> `ncu` pass are the pending confirmation that would turn "parity" into a signed
> number; until then the defensible claim is *at parity with SDPA on the
> HBM-bound shapes, and decisively past our own v3.*

Where v6 still trails (0.68–0.83×) is the **small, L2-resident** shapes
(`kv_len ≤ 1024` at these batch sizes fit in the 72 MB L2). There the kernel is no
longer HBM-bound — flash's L2 blocking and lower fixed overhead win, and our
combine pass plus split launch is a larger fraction of a much shorter runtime.
That is the honest boundary of this result and the next thing to chase; it is
recorded, not rounded away.

## What this does to the end-to-end story

`bench_decode_step.py --part e2e`, the locked workload (`batch=16, prompt=512,
generate=512`, greedy), each config correctness-gated against vanilla first:

| config | description | tok/s | vs vanilla | vs fair (`gqa`) |
|---|---|---|---|---|
| `vanilla` | DynamicCache + repeat_kv + SDPA | 343.0 | 1.00× | 0.64× |
| `gqa` | DynamicCache + SDPA(`enable_gqa`) | 533.1 | 1.55× | 1.00× |
| **`ours`** | DynamicCache + **v6 split-K kernel** | **538.6** | **1.57×** | **1.01×** |
| `taxfree` | StaticCache + SDPA(`enable_gqa`) | 155.0 | 0.45× | — (negative result) |
| — | vLLM 0.6.6 (Phase 0) | 703.2 | 2.05× | 1.32× |

The change from Phase 7 is small but it is the *right sign*: Phase 7's v3-hooked
`ours` measured **485 tok/s and lost to the `gqa` fair baseline (535)**. With v6,
`ours` is **538.6 — now marginally ahead of `gqa` (533.1)**. The custom attention
kernel is finally a net e2e positive rather than a net negative.

It is only *marginal* e2e, and Phase 7 already explained why: at `batch=16` the
projection GEMMs are **82% of the fair decode step** and stream 15.01 GB of fp16
weights each step at the memory roofline. Attention is a thin slice; making it
match SDPA instead of losing 4.55× moves the whole step by ~1%. **The attention
kernel is now at parity with SOTA; it is no longer the thing holding e2e back.**

And the vLLM gap is unchanged at **1.31×** — because, exactly as Phase 7 found,
that gap is *framework tax* (the `DynamicCache` `torch.cat` plus unfused
elementwise), not attention or GEMM kernel quality. The only lever that closes it
is breaking the 15.01 GB/step weight roofline, i.e. **W4A16** — the repo's
surviving thesis, untouched by this phase.

## What we learned

- **The diagnosis has to survive a fix, not just an argument.** Phase 7 argued v3
  was occupancy-bound from a roofline and a competitor's bandwidth. v6 is the
  falsifiable version of that argument: fix the occupancy, and if the diagnosis
  was right the bandwidth follows. It did — 18% → 82%. That is the difference
  between "we think it's occupancy" and "it was occupancy."
- **Split-K was never the wrong idea; the block was.** v4's revert concluded
  split-K doesn't help our workload. It concluded that from a single-warp block
  that couldn't have used the parallelism split-K delivers. Same technique, right
  geometry, opposite result.
- **Beating SOTA on the headline shape does not mean beating it everywhere.** v6
  wins where the kernel is HBM-bound and loses where it is L2-bound. A single
  "1.01×" would have hidden that; the sweep shows it. The `flash_attn`-comparison
  row that sat _TBD_ through all of Phase 1 is the one that made this legible.

## What's open

- **The L2-resident shapes.** v6 trails flash at `kv_len ≤ 1024`. Reducing the
  combine/launch overhead (fewer, fatter splits when the grid already fills the
  SMs; fusing the combine) and L2-aware chunking are the levers.
- **`ncu` with locked clocks.** Still the missing profiler confirmation — achieved
  occupancy and `dram__throughput.avg.pct_of_peak` for v6, and the exact remaining
  gap to flash on the small shapes.
- **Tensor-core / fp8 paths and the INT8/INT4 KV variants.** The INT8 and INT4
  KIVI decode kernels are still built on the v3 single-warp body; porting them onto
  the v6 split-K geometry should hand them the same occupancy win.

## Reproduction

```bash
scripts/lock_clocks.sh                                 # do this first
python benchmarks/bench_decode_step.py --part kernel   # v6 vs SDPA sweep
python benchmarks/bench_decode_step.py --part e2e      # per-config e2e
```
