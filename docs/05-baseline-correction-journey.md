# Phase 7 Journey — Baseline Correction

> Companion to [`results/RESULTS.md`](results/RESULTS.md) (numbers) and the
> per-track journey docs ([01](01-fused-attention-journey.md),
> [02](02-kv-cache-compression-journey.md),
> [03](03-quantized-matmul-journey.md),
> [04](04-end-to-end-integration-journey.md)).
>
> Phase 7 started as a question — *"how are we more than 2× slower than
> vLLM?"* — and ended by invalidating three of this repo's headline claims.
>
> The short version: **our PyTorch SDPA baseline was crippled.** Both the
> Phase 1 microbench and the Phase 4 end-to-end harness fed SDPA a
> 4×-expanded GQA key/value cache. Every "vs SDPA" and "vs HF" number in
> this repo was measured against that handicap. Corrected, the Phase 1
> attention kernel is **4.55× slower** than PyTorch, not 1.91× faster; the
> Phase 4b speedup belonged to a one-line framework fix rather than to the
> INT4 kernel; and vLLM's 2.09× has essentially nothing to do with kernels.
>
> The one result that *survives and strengthens*: the W4A16 track. Once the
> framework tax is removed, 82% of a decode step is the projection GEMM, and
> weight quantization is the only lever that breaks the memory roofline both
> we and vLLM are pinned to.

## Setting

- **GPU**: RTX 4090 (sm_89), 1008 GB/s peak HBM, 72 MB L2.
- **Model**: Llama 3.1 8B Instruct, FP16, `sdpa` attention.
- **Locked workload**: `batch=16, prompt=512, generate=512`, greedy, EOS
  suppressed — the same one [`bench_e2e.py`](../benchmarks/bench_e2e.py) uses.
- **New harness**: [`benchmarks/bench_decode_step.py`](../benchmarks/bench_decode_step.py).
- **Caveat on every number below**: measured with unlocked clocks on a
  machine running a desktop session. Run-to-run spread is a few percent;
  the effects reported here are 30–350%. `scripts/lock_clocks.sh` before
  these land as final.

---

## The question

`RESULTS.md` Phase 0 records HF `generate()` at 354.6 tok/s and vLLM 0.6.6 at
703.2 tok/s — vLLM is 1.98× the HF baseline. The natural reading, and the one
this repo implicitly adopted, is *vLLM has better kernels; our job is to close
the kernel gap.*

That reading is wrong, and the first profile says so immediately.

## Step 1 — profile the vanilla decode step

`batch=16`, `kv_len=544`, one decode step of the unmodified HF model:

| | per decode step |
|---|---|
| Wall time | 37.22 ms |
| **GPU-busy time** | **37.25 ms** |
| Host stall | **−0.03 ms (0%)** |
| Kernel launches | 1429 |

The GPU is busy 100% of the step. **There is no host-side bubble.** This
already contradicts the hypothesis in
[04's Phase 6 section](04-end-to-end-integration-journey.md) that ~5–10 ms/step
of Python dispatch overhead is what stopped the Phase 6 kernel from cashing in.
(That claim was made about the *W4A16-patched* model, so it is not strictly
refuted here — but on vanilla HF, host stall is exactly zero, which makes it a
poor prior. It should be measured before Phase 7 spends effort on CUDA graphs.)

Where the GPU time goes:

| Kernel | ms/step | What it is |
|---|---|---|
| `cutlass ... f16_16x16` | 19.13 | the projection GEMMs |
| `pytorch_flash::flash_fwd_kernel` | 7.54 | attention |
| `elementwise_kernel` | 6.71 | mostly the `repeat_kv` copy |
| `CatArrayBatchedCopy_contig` | 1.83 | `DynamicCache`'s `torch.cat` |
| everything else | ~2.0 | norms, RoPE, residuals |

The GEMMs stream 15.01 GB of fp16 weights in 19.13 ms — **785 GB/s, 78% of
peak.** They are already at the memory roofline. **The GEMMs are not the gap.**

The gap is the other 18 ms.

## Step 2 — name the tax

Three costs in the HF decode step have nothing to do with kernel quality, and a
serving engine does not pay them. Microbenched at the real shapes
(`batch=16, kv_len=544`, ×32 layers):

| Cost | ms/step | Why vLLM doesn't pay it |
|---|---|---|
| `repeat_kv` materialization | 8.22 | its attention kernel reads GQA natively |
| attention excess from that expansion | 4.70 | 5.62 ms expanded vs **0.92 ms** GQA-native |
| `DynamicCache`'s `torch.cat` | 1.54 | paged KV appends in place |
| unfused RMSNorm / RoPE / SiLU | ~3 | fused |

**`repeat_kv` is the villain.** transformers expands the 8 KV heads to 32
*before* calling SDPA, via `expand().reshape()` — and the reshape copies. Per
layer that turns 35.7 MB of KV into 142.6 MB. Across 32 layers it turns 1.14 GB
of KV reads per step into 4.56 GB, and adds a 4.56 GB write to build the copy.

So the decode step is `19.1 ms of real work + ~14.4 ms of tax + ~3.7 ms of
other`. Almost exactly 2×. **vLLM's 2.09× is the tax.**

## Step 3 — build the honest baseline

[`bench_decode_step.py`](../benchmarks/bench_decode_step.py) runs the locked
workload under four configs, each removing one tax, correctness-gated against
vanilla first (max `|Δlogit|` ≤ 3.5e-2, all pass):

| config | description | ms/step | tok/s | vs vanilla | weight BW |
|---|---|---|---|---|---|
| `vanilla` | DynamicCache + repeat_kv + SDPA | 44.15 | 344.8 | 1.00× | 340 GB/s |
| **`gqa`** | DynamicCache + SDPA(`enable_gqa=True`) | **27.62** | **535.2** | **1.55×** | 544 GB/s |
| `static` | StaticCache + SDPA(`enable_gqa=True`) | 99.69 | 156.7 | 0.45× | 151 GB/s |
| `ours` | DynamicCache + Phase 1 v3 decode kernel | 30.70 | 485.4 | 1.41× | 489 GB/s |
| — | vLLM 0.6.6 (Phase 0 number) | — | 703.2 | 2.04× | — |

`vanilla` reproduces the Phase 0 HF number (344.8 vs 354.6, ~3%), which
validates the harness.

**`gqa` is the honest denominator for every kernel claim in this repo.** It is
stock PyTorch with `repeat_kv` neutered — no CUDA of ours involved. It is
1.55× vanilla.

### `static` — a negative result worth keeping

Preallocating the cache does remove the `torch.cat`. But on transformers 4.47 +
torch 2.5, `StaticCache` forces a full `[B, 1, 1, max_cache_len]` float mask
into SDPA, which disqualifies the flash backend. SDPA falls back to math —
`gemvx` plus an explicit masked softmax — and attends over all 1024 preallocated
slots from step one. Profile confirms: **no `flash_fwd` kernel at all**, 95.35 ms
of GPU-busy time, 1916 launches.

Removing the `cat` (worth only ~1.7 ms/step) is a win only with a cache that
still hands SDPA a contiguous live prefix, or with a paged attention kernel that
takes a length argument. That is exactly what vLLM built and we did not.

## Step 4 — the Phase 1 kernel, measured honestly

Both the Phase 1 microbench and the reference implementation expand GQA before
calling SDPA (`reference/attention_ref.py:94`, `_expand_gqa`). So the famous
"PyTorch SDPA = 1.36 ms" baseline is SDPA handed a 4×-expanded cache.

Sweeping our v3 kernel against both baselines:

| batch | kv_len | sdpa + `repeat_kv` | sdpa `enable_gqa` | ours (v3) | ours vs gqa |
|---|---|---|---|---|---|
| 16 | 512 | 356.3 µs | 29.7 µs | 63.5 µs | 0.47× |
| 16 | 768 | 573.4 µs | 37.2 µs | 93.2 µs | 0.40× |
| 16 | 1024 | 700.4 µs | 41.9 µs | 115.6 µs | 0.36× |
| 16 | 2048 | 1375.2 µs | 156.8 µs | 379.7 µs | 0.41× |
| 8 | 1024 | 340.1 µs | 28.5 µs | 114.7 µs | 0.25× |
| **8** | **4096** | **1367.0 µs** | **157.7 µs** | **717.8 µs** | **0.22×** |

The last row is Phase 1's own reference workload. The 1367 µs column *is* the
1.36 ms baseline, to three digits.

**Against SDPA's real GQA path, v3 is 4.55× slower, not 1.91× faster.**

### Why — and why the v4 split-K revert was misdiagnosed

Torch dispatches `flash_fwd_splitkv_kernel`: FlashDecoding, split-K over the
sequence. That is precisely the technique
[Phase 1 explored and reverted](01-fused-attention-journey.md), concluding "v3
wasn't actually grid-limited at our workload … the HBM/L2-throughput limit …
bandwidth was the ceiling."

Check the arithmetic. At `batch=8, kv_len=4096` the kernel streams 128 MB:

- flash: 128 MB / 157.7 µs = **812 GB/s — 81% of peak**
- v3: 128 MB / 717.8 µs = **178 GB/s — 18% of peak**

Bandwidth was never the ceiling. Flash reaches 81% of it on the same GPU, same
workload, same bytes. **v3 is occupancy-bound, not bandwidth-bound** — the v3
step itself records the trade it made ("per-SM occupancy drops 48→16 warps
(full→33%) because each block has only 1 warp"), then the v4 step attributed the
resulting split-K regression to a bandwidth ceiling that flash demonstrably is
not hitting.

The split-K *idea* was right. Our split-K *implementation* lost to a design that
had already given away two thirds of its occupancy. Layering split-K on top of a
single-warp block adds launch and combine cost without unlocking the parallelism
the block geometry threw away.

## Step 5 — what this does to Phases 4b and 4c

**4b's 1.55× was never the INT4 kernel.** Phase 4b measured 521.7 tok/s. Plain
fp16 `enable_gqa` measures 535.2 tok/s — the same number, slightly better, with
no quantization. 4b's `patched_int4_decode_attention` replaces
`LlamaSdpaAttention.forward` and therefore skips `repeat_kv`; that is where the
1.55× came from. The INT4 KIVI cache bought Δppl +0.196 and a greedy match rate
of 0.5047, and returned nothing on throughput.

This does **not** invalidate Phase 2. The KV memory result (0.27× bytes) and the
KIVI accuracy finding (per-channel K is 2.36× better than per-token K at INT4)
stand on their own. What changes is that 4b is a **memory** result, not a speed
result, and must be presented as one.

**4a's 1.025× is explained.** `integration/attention_patch.py:61` rebinds
`F.scaled_dot_product_attention`, which HF calls *after* `repeat_kv` has already
run. So 4a pays the full 8.2 ms expansion, then pays a second copy
(`key[:, ::n_rep].contiguous()`) to undo it, and only recovers attention kernel
time — with a kernel that is slower than the one it replaced. Hooking it
correctly (config `ours`, 1.41×) still loses to doing nothing at all (`gqa`,
1.55×).

**4c was graded on the wrong curve.** Phases 5 and 6 kept finding that W4A16
kernel wins "didn't translate e2e." They were measured against a 44 ms step in
which the GEMM was only 43% of the time — a 13 ms kernel saving disappeared into
24 ms of tax. Rebased on the fair step, the GEMM is **20.27 of 24.86 ms = 82%.**

## What survives, and where the prize is

Against the fair baseline, vLLM's lead is **703.2 / 535.2 = 1.31×**, and the
residual is the `cat` plus unfused elementwise. Our GEMMs run at 740 GB/s;
vLLM's implied rate on the same 15.01 GB of weights is ~715 GB/s.

**vLLM has no fp16 kernel advantage at this batch size.** It has a better
framework.

Which means the only way to actually beat it is to stop reading 15.01 GB of
weights per step. That is Track 3, and the arithmetic is now unambiguous:

| | weight bytes/step | GEMM ms @ 740 GB/s | step ms | tok/s |
|---|---|---|---|---|
| fp16 (us and vLLM) | 15.01 GB | 20.3 | 24.9 | 535–703 |
| W4A16 at parity efficiency | 3.75 GB | 5.1 | ~9.7 | **~1460** |

Even at half that efficiency, W4A16 beats vLLM-fp16 outright. **The repo's
original thesis was right; only its baseline was wrong.**

## What we learned

- **A baseline you didn't profile is a baseline you don't understand.** Every
  number in Phases 1, 2, and 4 was correct as arithmetic and wrong as a claim,
  because the denominator carried a 4× handicap nobody looked at. The bug was
  four lines up from the kernel, in `_expand_gqa`.
- **"We're 2× slower than vLLM" was a framework fact, not a kernel fact.** It
  took one profiler run to see and would have redirected two phases of work.
- **A negative result about your own conclusion is worth more than a positive
  result about your kernel.** The v4 split-K revert was reasoned carefully from
  a bandwidth ceiling that a competing kernel exceeds by 4.5×. The reasoning was
  internally consistent and externally false. Only a SOTA comparison caught it —
  which is exactly what the `flash_attn` row in `RESULTS.md`, still marked
  _TBD_ since Phase 1, was supposed to be for.
- **Fill in the SOTA row first, not last.** It is the only cell in the table
  that can tell you the rest of the table is meaningless.

## What changes

- `RESULTS.md`, `README.md`, and `01-fused-attention-journey.md` carry
  corrections pointing here. The original numbers are preserved with their
  handicap named, not deleted.
- `gqa` replaces `vanilla` as the denominator for all kernel claims.
- Phase 1's "1.91× over SDPA" is retired. The kernel stays on `main` as a
  correctness and learning artifact; it is not a performance win.
- Phase 4b is reframed as a memory result.

## What's open

- **Fix v3's occupancy, then revisit split-K.** Multi-warp blocks + split-K over
  the sequence is what flash does. Whether we can approach 812 GB/s is the real
  Phase 1 question, and it was never asked.
- **W4A16 at M=16 against Marlin**, not against vLLM-fp16. Kernel-to-kernel,
  same math. This is the only path to beating a serving engine from a research
  harness, and Phase 6's two identified optimizations (dequant into MMA register
  layout; double-buffered shmem) are the concrete next steps.
- **Measure host stall on the 4c path** before investing in CUDA graphs. On
  vanilla it is zero.
- **`ncu` profile with locked clocks.** Still pending, still the thing that
  would have caught this in Phase 1.

## Reproduction

```bash
scripts/lock_clocks.sh                             # do this first
python benchmarks/bench_decode_step.py --part kernel   # Step 4 table
python benchmarks/bench_decode_step.py --part e2e      # Step 3 table
```
