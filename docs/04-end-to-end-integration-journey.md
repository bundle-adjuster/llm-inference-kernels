# Phase 4 Journey — End-to-End Integration

> Companion to [`results/RESULTS.md`](results/RESULTS.md) (numbers) and the
> per-track journey docs ([01](01-fused-attention-journey.md),
> [02](02-kv-cache-compression-journey.md),
> [03](03-quantized-matmul-journey.md)).
>
> Phase 4 takes the three kernels built in Phases 1–3 and integrates them
> into the actual Llama 3.1 8B Instruct model via HF monkeypatch, one
> kernel at a time, with a full accuracy/latency/memory eval after each
> step. The point isn't another kernel — it's making the tradeoff visible
> on the *real model* across the metrics that matter (MMLU/HellaSwag/ARC-C
> via lm-evaluation-harness, plus WikiText-2 PPL, greedy-token match,
> decode tokens/sec, and peak VRAM).
>
> The headline lesson of Phase 4: **kernel-level wins don't all translate
> 1:1 to the integrated model**, and the gaps are themselves the
> interesting result. 4a's bit-identical decode delivers only +2.5% on the
> batched workload because attention is a small fraction of decode time;
> 4b's KIVI KV cache delivers 1.55× because it moves what was a dependency
> chain into a saved chain; 4c's W4A16 delivers 51% memory reduction but
> a tok/s *regression* at batch=16 — and the cause is a kernel-design
> assumption (M=1) that the e2e workload violates.

## Setting

- **GPU**: RTX 4090 (sm_89), 1008 GB/s peak HBM, 72 MB L2.
- **Target model**: Llama 3.1 8B Instruct, FP16 weights, `sdpa` attention.
- **Locked workload** (end-to-end metrics):
  `batch=16, prompt=512, generate=512` greedy with EOS suppressed.
  This is what [`benchmarks/workload.py`](../benchmarks/workload.py)
  defines as `E2E_*`; matches Phase 0's HF/vLLM baselines.
- **Eval bar** (per-step, on the patched model):
  - **Standard accuracy** via [lm-evaluation-harness][lm-eval]:
    MMLU 5-shot, HellaSwag 0-shot (acc_norm), ARC-Challenge 25-shot
    (acc_norm). This is what the GPTQ/AWQ/KIVI papers report.
  - **End-to-end**: WikiText-2 PPL (64 chunks × 2048 tok), greedy-match
    rate vs vanilla on 10 fixed prompts (integration-bug guard),
    decode tokens/sec, peak VRAM.
- **Baseline** numbers (vanilla HF Llama 3.1 8B Instruct, FP16, sdpa):
  - MMLU **68.32%** · HellaSwag (acc_norm) **79.51%** · ARC-C (acc_norm) **60.84%**
  - WikiText-2 PPL **7.055** · greedy_match **1.000**
  - **335.8 tokens/sec · peak VRAM 18.50 GB**
  - These are within ~1 pp of Meta's published numbers (69.4 / 80.4 / 60.3),
    a sanity check on the harness setup.

[lm-eval]: https://github.com/EleutherAI/lm-evaluation-harness

The integration approach for every step is **monkeypatch HF transformers
on a loaded model** — we never write a new model class or fork
transformers. Three reasons: (1) it isolates the kernel work from the
serving harness, (2) it lets `model.generate()` continue handling sampling,
KV-cache lifecycle, and tokenization, (3) it leaves a clean A/B between
vanilla and patched in the same process.

---

## 4-prep — Eval infrastructure · commit `1c56d82`

### Theory

Before patching anything, we need the apples-to-apples eval setup that the
4a/4b/4c rows will compare against. The KIVI and W4A16 papers reach their
accuracy numbers via lm-evaluation-harness on the same three tasks, so
we use the same. End-to-end metrics (PPL, tok/s, peak VRAM, greedy match)
get their own runner.

### Design choices

- **Two entry-point scripts**:
  - `scripts/run_lm_eval.py` runs MMLU/HellaSwag/ARC-C and writes
    `docs/results/lm_eval/{config}.json`.
  - `scripts/run_e2e_eval.py` runs PPL + greedy match + tok/s + peak VRAM,
    writes `docs/results/e2e_eval/{config}.json`.
  - Both accept a pre-loaded HF model so the 4a/4b/4c orchestrator can
    pass a *patched* model in without re-loading.
- **Greedy-match reference**: the vanilla run generates 10 fixed prompts ×
  64 new tokens with greedy sampling and saves the token IDs to
  `docs/results/e2e_eval/vanilla_reference_outputs.json`. 4a/4b/4c
  compare against that file. A mismatch points at an integration bug; a
  bit-identical run (e.g. 4a) confirms the kernel is doing what its
  Phase 1 microbench said.
- **`batch_size="auto"`** in HFLM. MMLU's 5-shot contexts blow OOM at
  `batch_size=4`; auto tunes per task.
- **One process per run.** Both eval scripts load the model fresh — slower
  but avoids state leaks between configs, and we never measure two
  configs in the same Python process.

### Result

Vanilla Llama 3.1 8B Instruct baseline (the "starting line" for 4a/4b/4c):

| Metric | Value |
|---|---|
| MMLU (5-shot) | 68.32% |
| HellaSwag (0-shot, acc_norm) | 79.51% |
| ARC-Challenge (25-shot, acc_norm) | 60.84% |
| WikiText-2 PPL | 7.055 |
| HF tokens/sec | 335.8 |
| Peak VRAM | 18.50 GB |

The lm-eval-harness numbers landed within ~1 pp of Meta's published values
(69.4 / 80.4 / 60.3) — small gap is expected (different few-shot order,
slight tokenization differences, FP16 instead of BF16 base). Sanity check
passes; we trust the harness for everything below.

### What we learned

- **Pre-loaded-model wrap is non-trivial.** lm-eval-harness defaults to
  loading from a string `pretrained=<id>`; passing a patched model needs
  `HFLM(pretrained=model, tokenizer=tokenizer, batch_size=...)`. Without
  this we'd have to re-implement the eval harness or fork it.
- **Greedy-match is a *quiet* but powerful test.** PPL on a fixed forward
  pass is forgiving of single-token logit drift; greedy-match makes
  every-token equality the test. 4a's `greedy_match_rate=1.0` was the
  single most reassuring number in Phase 4.

---

## 4a — Attention kernel integration · commit `d6d8c88`

### Theory

Phase 1 v3 `decode_attention` was a 1.91× win over PyTorch SDPA on the
microbench (batch=8, seqlen_kv=4096). Question: does that 1.91× show up
end-to-end? Two reasons it might not:
1. Attention is only part of decode-step time. At batch=16 with avg
   kv_len≈768, attention is much smaller than the MLP/QKV projection
   GEMMs.
2. HF's `LlamaSdpaAttention.forward` runs `repeat_kv` before calling
   `F.scaled_dot_product_attention`, expanding K/V from
   `[B, n_kv_heads=8, S, D]` to `[B, n_heads=32, S, D]`. Our kernel
   expects the GQA shape; if we don't undo the expansion, we read 4×
   the KV bandwidth and lose the whole reason we built it.

### Design choices

- **Patch surface**: rebind `F.scaled_dot_product_attention`. HF looks up
  the symbol on the module object at call time, so a single assignment
  intercepts every layer's attention. No subclassing of `LlamaSdpaAttention`
  needed.
- **Dispatch by `q_len`**: q_len == 1 → decode (our kernel); q_len > 1 →
  prefill (original SDPA). Cheap and clean.
- **Un-expand `repeat_kv`**: `K[:, ::n_rep].contiguous()`. `repeat_kv` is
  exactly `repeat_interleave` along dim 1, so taking every `n_rep`-th
  head recovers the original GQA tensor. The `.contiguous()` copies
  ~33 MB at the e2e workload — ~33 µs at HBM bandwidth, well under the
  kernel's own runtime.
- **Output reshape**: `out.unsqueeze(2)` back to `[B, n_heads, 1, D]`.

### Result

| Metric | Vanilla | 4a | Δ |
|---|---|---|---|
| MMLU | 68.32% | **68.32%** | 0.00 pp |
| HellaSwag acc_norm | 79.51% | **79.51%** | 0.00 pp |
| ARC-C acc_norm | 60.84% | **60.84%** | 0.00 pp |
| PPL | 7.055 | **7.055** | 0 |
| greedy_match | 1.000 | **1.000** | 0 |
| tokens/sec | 335.8 | **344.1** | **+2.5%** |
| peak VRAM | 18.50 GB | 18.57 GB | +0.07 GB |

**Bit-identical accuracy** on every prefill-based metric — because prefill
goes through original SDPA — and **greedy_match=1.0** confirms decode is
bit-perfect too (across 10 × 64 = 640 generated tokens). The +0.07 GB
peak VRAM is the un-expansion buffer (~66 MB).

The **+2.5% tok/s** is the surprise — much smaller than the microbench's
1.91×. The reason is structural: attention is only ~4% of decode time at
the e2e workload (batch=16, avg kv_len≈768). 50 MB/layer KV bandwidth ÷
1 TB/s ≈ 50 µs/layer × 32 layers = 1.6 ms per decode step. The full
decode step is ~41 ms; the rest is projection GEMMs + lm_head. Halving
attention saves <1 ms / 41 ms ≈ 2%.

### What we learned

- **A microbench win is local to its workload.** Phase 1 v3 was 1.91×
  faster *on attention alone*. End-to-end, attention is a fraction of the
  per-decode-step budget; speeding it up 2× yields a small total speedup.
  This is the standard "Amdahl says hi" outcome — and it's why the *next*
  step (4b) was structurally able to do much better, by attacking a
  bigger fraction of the step.
- **GQA awareness has to be respected at every layer.** HF's
  `repeat_kv`-before-SDPA is a serving-stack convention; our kernel
  assumes the pre-`repeat_kv` shape. The slice-and-`.contiguous()` un-do
  was the cheapest fix that preserved correctness. The more invasive
  alternative — subclass `LlamaSdpaAttention.forward` to skip `repeat_kv`
  — would also save the slice copy, but Phase 4a isn't where the bytes
  matter.
- **Test parity vs vanilla matters more than headline speedup at this
  step.** `greedy_match=1.0` is what told us the patch is correct. A
  faster but slightly different output would be a regression, not a
  speedup.

---

## 4b — INT4 KIVI KV-cache integration · commit `0836a77`

### Theory

Phase 2c built `decode_attention_int4`, which reads a KIVI-style packed
INT4 K/V cache (per-channel groupwise K with `group_size=32` along seqlen,
per-token V) and computes attention without ever materializing fp16
K/V. Kernel-level: **1.29× faster than v3 fp16 attention, Δppl +0.196
on WikiText-2** (Phase 2c). 4b's job is to plumb a *real* INT4 cache into
HF Llama and exercise it end-to-end. Three sub-questions:

1. **Cache mechanics.** HF's `Cache.update()` returns fp16 K/V for SDPA
   consumption. How do we plumb int4 storage without breaking that contract?
2. **Prefill vs decode.** Prefill uses fp16 attention (SDPA); decode wants
   the int4 kernel. Both paths need to read the same cache.
3. **lm-eval-harness.** HFLM doesn't pass `past_key_values` through, so
   the cache won't be exercised on log-likelihood evals. How do we surface
   the KIVI accuracy hit there?

### Design choices

- **`Int4KIVICache(transformers.Cache)`** (`integration/kv_int4_cache.py`).
  Stores K packed-int4 + per-channel groupwise scales (group=32) and V
  packed-int4 + per-token scales — the Phase 2c scheme.
- **Residual buffer for partial groups.** Per-channel K quantization with
  group_size=32 needs 32 tokens to compute each scale. Decode appends 1
  token at a time, so until 32 accumulate, we hold them as fp16 in a
  residual buffer per layer. Once the residual hits 32, the chunk is
  quantized into packed storage. Max residual size = 32 tokens × 32 layers
  × ~16 KB ≈ 32 MB — small.
- **Two return paths from `update()`**:
  - `update(K, V, ...)`: HF Cache contract — append + return dequantized
    fp16 (quantized portion + residual concatenated) for the SDPA path
    during prefill.
  - `append_only(K, V, ...)`: the storage half of `update` minus the
    fp16 materialize. Used by the decode fast path which reads the
    cache via `get_quantized_for_attention()` instead.
- **`LlamaSdpaAttention.forward` replacement for the int4 decode path**
  (`patched_int4_decode_attention` in `integration/attention_patch.py`):
  - q_len > 1 (prefill) → fall back to the original forward; `cache.update`
    returns KIVI-noisy fp16 K/V to SDPA.
  - q_len == 1 (decode) + `isinstance(cache, Int4KIVICache)` → project
    q/k/v, RoPE, `cache.append_only`, `cache.get_quantized_for_attention`,
    call `decode_attention_int4` directly on the packed tensors. Bypasses
    `repeat_kv` *and* SDPA entirely.
- **F.sdpa rebind for lm-eval-harness** (`patched_kivi_int4_sdpa`).
  Same KIVI quantize-and-dequantize math, applied directly to K/V before
  the original SDPA. Identical noise pattern to the cache path → consistent
  numbers across e2e and lm-eval.

### Result

| Metric | Vanilla | 4b | Δ |
|---|---|---|---|
| MMLU | 68.32% | 67.29% | **-1.03 pp** |
| HellaSwag acc_norm | 79.51% | 79.07% | -0.44 pp |
| ARC-C acc_norm | 60.84% | 61.43% | +0.59 pp (noise) |
| PPL | 7.055 | 7.256 | **+0.20** |
| greedy_match | 1.000 | 0.5047 | -0.50 |
| tokens/sec | 335.8 | **521.7** | **1.55×** |
| peak VRAM | 18.50 GB | 18.41 GB | -0.09 GB |

The **PPL delta of +0.20 matches Phase 2c's kernel-level +0.196 to within
rounding** — direct validation that the integration faithfully reproduces
the kernel's noise. The MMLU drop of ~1 pp is the expected hit for
INT4 KV at group_size=32; consistent with the KIVI paper's numbers for
similar bit-widths.

**tokens/sec 1.55×** — bigger than Phase 2c's 1.29× kernel-level win,
because the forward replacement *also* skips `repeat_kv` and the SDPA
dispatch overhead on top of the int4 kernel speedup itself.

**`greedy_match_rate` 0.50** is informative but not concerning: small
per-token logit noise from KIVI flips argmax choices a few tokens in, and
greedy decoding compounds those flips. PPL is the meaningful next-token-
prediction metric (and it only moved ~2.8%).

**Peak VRAM unchanged.** The INT4 cache really does store ~0.27× the
bytes of fp16 (Phase 2 storage tests proved this and 4b uses the same
quantize kernels) — but peak is dominated by prefill, where
`cache.update()` materializes the full fp16 K/V via `_materialize_fp16`
for the SDPA path. Steady-state decode-only memory would show the drop;
the e2e workload's prefill phase hides it.

### What we learned

- **Same noise math, two patch surfaces.** lm-eval and e2e take separate
  context managers (F.sdpa rebind vs forward replacement) but call the
  same KIVI quantize CUDA kernels. The PPL numbers agreeing across both
  paths is the cross-check that says "yes, these surfaces are
  equivalent." Don't try to share state between them — keep the patches
  disjoint and let them agree by construction.
- **The greedy/PPL split has real meaning.** PPL measures the *probability
  mass* on the right next token; greedy match measures the *argmax* on
  the right next token. KIVI moves PPL by ~2.8% but moves greedy match by
  ~50% because argmax is a much more brittle function of small logit
  perturbations. Knowing which metric to trust for which question is part
  of reading these tables.
- **Where peak VRAM hides.** Even with INT4 cache storage proven, the
  e2e prefill peak is what RESULTS.md records, and that peak is fp16-
  materialize-dominated. The honest framing isn't "INT4 KV saves no
  memory" — it's "the prefill *spike* doesn't, but steady-state
  long-running decode does." A decode-only INT4 kernel for prefill (and
  the matching update path) would close this; out of Phase 4 scope.

---

## 4c — W4A16 weight integration · commit `8882880`

### Theory

Phase 3c built `decode_attention`'s GEMM analogue: an int4 weight + fp16
activation GEMM that's **2.88× / 6.97× / 2.96× over fp16 cuBLAS** on
Llama 3's three Linear shapes (QKV/O, MLP up/gate, MLP down) **at M=1**.
Phase 3 explicitly scoped to M=1: "batched-decode M>1 is integration
concern (Phase 4)" — and 4c is where that comment cashes in.

Two things 4c needs to demonstrate:

1. **Weight memory win.** Llama 3.1 8B is ~16 GB in fp16. Quantizing the
   projections to packed INT4 should drop weight storage to ~5.7 GB
   (the rest is embed + lm_head, intentionally fp16).
2. **Decode latency.** With the projections taking most of the per-decode-
   step time at this workload, the Phase 3c kernel should — at its design
   point — dominate the speedup story.

### Design choices

- **`QuantizedLinear(nn.Module)`** (`integration/w4a16_patch.py`):
  - Stores packed INT4 weights `[K/8, N] int32` + per-channel groupwise
    scales `[n_groups, N] fp16`, group_size=128 along K.
  - Forward dispatches on `M = act.shape[:-1].numel()`:
    - `M < 256` (decode and small batched-decode): `llmik_cuda.w4a16_gemm`
      — routes to Phase 3c's decode-optimized kernel at M=1, Phase 3b's
      naive at M>1.
    - `M ≥ 256` (prefill): unpack int4 → multiply by scale → fp16 weight,
      then `torch.matmul` (cuBLAS). The naive kernel's M-cost crosses
      cuBLAS around M ~ 256 on this GPU; above that, dequant-then-cuBLAS
      is faster than the kernel.
- **One-way patch.** `patch_model_w4a16(model)` walks the 32 decoder
  layers and replaces all 7 Linears per layer (224 total) in place.
  The original fp16 weights drop their references and free on the next
  GC pass / `empty_cache` — without this, the fp16 weights would stay
  resident and we'd see *no* memory savings (this was the first bug I
  hit; the `handles`-list-for-restore approach kept the originals
  alive).
- **`lm_head` and `embed_tokens` stay fp16.** Big tensors but only run
  once per sequence (lm_head once at the head, embed once at input). No
  decode dominance, and quantizing them isn't standard practice.
- **Reuse 4b's patches for the rest.** Step=4c stacks: W4A16 patch +
  Int4KIVICache + `patched_int4_decode_attention` + `patched_kivi_int4_sdpa`
  for lm-eval. Patches compose cleanly because they target different
  module classes / function tables.

### Result — memory win, latency regression (the lesson)

| Metric | Vanilla | 4c | Δ |
|---|---|---|---|
| MMLU | 68.32% | 62.40% | **-5.92 pp** |
| HellaSwag acc_norm | 79.51% | 77.51% | -2.00 pp |
| ARC-C acc_norm | 60.84% | 55.72% | -5.12 pp |
| PPL | 7.055 | 8.087 | **+1.03** |
| greedy_match | 1.000 | 0.2672 | -0.73 |
| **tokens/sec (B=16, locked)** | **335.8** | **40.9** | **0.12× (regression)** |
| tokens/sec (B=1, single-stream) | 48.9 | **56.9** | **1.16×** |
| **Peak VRAM** | **18.50 GB** | **9.05 GB** | **-9.45 GB (-51%)** |

**Memory: large and immediate.** After `patch_model_w4a16`, weight VRAM
goes from 16.06 GB to 5.70 GB — **10.36 GB saved on weights**, exactly
what we set out to demonstrate. Peak VRAM during a full generate call
drops from 18.50 GB to **9.05 GB**, a 51% reduction. The headline memory
result of Phase 4.

**Accuracy hit compounds with 4b's KIVI noise.** MMLU -5.92 pp is in the
range modern PTQ papers report for similar bit-widths *without* recovery
techniques (GPTQ, AWQ both narrow this gap meaningfully). The PPL delta
matches what one would predict from compounding W4A16 + KIVI without
either being calibrated to the model.

**Tokens/sec at batch=16: regression, and the cause is exactly what
Phase 3 RESULTS forecast.** Walking through it:

- The locked workload is `batch=16`. At decode, each Linear sees activation
  `[B, 1, K] → [16, K]`, so **M=16**.
- `llmik_cuda.w4a16_gemm` at M=1 uses the Phase 3c decode-optimized kernel
  (which beat fp16 cuBLAS by 2.88-6.97× on Phase 3 shapes). At M>1, the
  launcher falls back to the Phase 3b naive kernel.
- The naive kernel runs the M=1 logic in a loop over M output rows. Each
  iteration re-reads the entire packed weight matrix. At M=16 we move
  **16× the weight bytes** that cuBLAS at M=16 moves; even though our
  weights are *half the size* (int4 vs fp16), we end up reading
  **4× more bytes** than vanilla per Linear forward.
- Decode latency is dominated by these reads. 4× weight bandwidth → ~4×
  slower per decode step → ~0.12× the tok/s vanilla delivers at batch=16.

**Tokens/sec at batch=1 is the proof that the kernel itself is fine.**
At batch=1 the M=1 decode-optimized kernel actually runs — and 4c is
1.16× over vanilla. The shape of the win matches Phase 3c's microbench
result. The kernel is *correct and fast at its design point;* the e2e
workload simply isn't that point.

### What we learned

- **An M=1 kernel doesn't extend for free to batched-decode.** The
  "fast path for decode" assumes a single output row — a single
  activation broadcast across a single tile of N. Stacking M output rows
  into the same block (so the weight read is amortized across M) requires
  a different inner loop. The naive M>1 fallback is correct but doesn't
  amortize — every M row re-reads the full int4 weight.
- **What production W4A16 kernels do differently (Marlin, AWQ-V2, GPTQMarlin
  in vLLM).** All of them have an M-aware fast path with an output tile
  `[BLOCK_M, BLOCK_N]` where `BLOCK_M ∈ [1, 16, 32, 64]`. Weights for the
  tile load *once* into shared memory; matmul produces BLOCK_M × BLOCK_N
  output values via tensor cores. The weight bandwidth amortizes across
  BLOCK_M — exactly the property our naive kernel lacks. Building that
  is the next step (Phase 5 below).
- **Phase 3 didn't oversell.** Phase 3 RESULTS row for 3c explicitly says
  "Fast path applies only at M=1. M>1 falls back to the 3b naive kernel
  via the launcher dispatch — 3c is decode-specialized; batched-decode
  M>1 is integration concern (Phase 4)." When we hit 4c, that comment was
  the diagnosis. The lesson here is structural, not "I broke something" —
  the kernel does what its design said it does.
- **Memory savings are the cleanest Phase 4 win.** Across 4a/4b/4c, the
  thing that compounded most reliably was peak VRAM: 18.50 → 18.57 →
  18.41 → 9.05 GB. The latency story has caveats (small win on 4a,
  meaningful on 4b, regression on 4c-at-batch=16); the memory story
  does not.

---

## Summary

The end-to-end picture across vanilla → 4a → 4b → 4c at the locked
workload (batch=16, prompt=512, generate=512):

| Step | tok/s | Peak VRAM | MMLU | HellaSwag | ARC-C | PPL | greedy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla HF | 335.8 | 18.50 GB | 68.32% | 79.51% | 60.84% | 7.055 | 1.000 |
| + 4a attn | 344.1 (+2.5%) | 18.57 | 68.32% | 79.51% | 60.84% | 7.055 | 1.000 |
| + 4b kv-int4 | **521.7 (1.55×)** | 18.41 | 67.29% (-1.0pp) | 79.07% (-0.4pp) | 61.43% | 7.256 (+0.20) | 0.50 |
| + 4c w4a16 | 40.9 (0.12×) | **9.05 (-51%)** | 62.40% (-5.9pp) | 77.51% (-2.0pp) | 55.72% (-5.1pp) | 8.087 (+1.0) | 0.27 |
| (4c at B=1) | 56.9 (1.16× vs van B=1) | — | — | — | — | — | — |

### Lessons we want to carry forward

1. **Kernel-level wins translate to e2e wins in proportion to the fraction
   of time the kernel was on.** 4a's 1.91× microbench attention speedup
   showed up as +2.5% e2e because attention is only ~4% of decode time
   at this workload. 4b's 1.29× microbench attention+KV speedup showed
   up as 1.55× because the same change *also* moved the dependency chain
   structure (Phase 2c's lesson) and *also* skipped HF's `repeat_kv` +
   SDPA dispatch overhead. The microbench number is necessary, not
   sufficient.
2. **Memory wins are sticky and immediate; latency wins depend on
   workload alignment.** The 51% peak VRAM cut at 4c is the most
   robust result in Phase 4 — it doesn't depend on M, on batch, on
   sequence length, or on whether prefill or decode is running. The
   latency wins do depend on all of those.
3. **The accuracy story compounds, sometimes uncomfortably.** 4b alone:
   MMLU -1 pp. 4c alone (W4A16, KV stays fp16): would be a few pp on
   its own. Stacking both: MMLU -5.9 pp. PTQ techniques like GPTQ /
   AWQ would close the W4A16 portion; we didn't build those. The honest
   framing for an interview: "stacking uncalibrated 4-bit weight + 4-bit
   KV without recovery is at the upper end of acceptable accuracy loss
   for this model — the path to closing the gap is well-known (calibration-
   aware quant), just outside our scope."
4. **An M=1 kernel is a single point in a 2-D design space.** The
   `[M, N]` GEMM design space has two axes; optimizing one and assuming
   the other generalizes is the mistake the Phase 4c regression
   illuminates. The fix isn't to back off the integration — it's to
   build the M-aware variant (Phase 5).
5. **Greedy-match is a fragile metric, but a fragile metric is useful
   *as an integration check*.** When `greedy_match=1.0` after 4a, we
   knew the patch was bit-exact without reading another metric. When it
   dropped to 0.50 at 4b we knew KIVI noise was actually being injected.
   When it dropped to 0.27 at 4c we knew the compounded quantization
   was actually changing logits at decode. It's the canary that says
   "the patch is doing what you intended" before any of the slower
   evals run.

## Phase 5 addendum — batched-decode W4A16 kernel

The Phase 4c regression had a designed-in fix; Phase 5 builds it.

### What landed

`w4a16_gemm_batched_decode_kernel` in `kernels/quant/quant_matmul.cu` —
same K-split-across-warps pattern as the Phase 3c decode kernel, but each
thread now accumulates a `BLOCK_M=16`-length vector of fp32 partials
instead of a scalar. Each warp has its own `[BLOCK_M, group_size]`
activation tile in shmem, loaded once per K group and reused across all
inner-loop iterations. The launcher routes:

```
M == 1        → Phase 3c decode kernel  (unchanged)
M ∈ [2, 16]   → Phase 5 batched-decode kernel
M  > 16       → Phase 3b naive (prefill catch-all)
```

### Result — recovery, not a full win

| Phase 4c at batch=16 | tok/s | vs vanilla HF | Note |
|---|---:|---:|---|
| M=1 kernel only (commit `8882880`) | 40.9 | 0.12× | M=16 falls to v0 naive: 16× weight read amplification |
| **+ Phase 5 batched-decode kernel** | **199.9** | **0.60×** | **4.9× recovery; still loses to cuBLAS (tensor cores)** |
| Reference: vanilla HF at B=16 | 335.8 | 1.00× | cuBLAS fp16 GEMM with tensor cores |
| Reference: 4c at B=1 (kernel design point) | 56.9 | 1.16× vs vanilla B=1 | M=1 kernel — unaffected by this change |

The new kernel is **correct** (max abs err 0.25, mean rel err ~0.001 —
the same fp16 reduction-order noise floor as 3c) and **amortizes weight
bandwidth as designed**: from M=2 through M=16, per-call time is
near-flat at ~200 µs/call (the work is BLOCK_M-sized regardless of
actual M ≤ BLOCK_M). The 4c memory savings (51% peak VRAM) and accuracy
metrics are unchanged — only the latency moved.

### Why we don't beat cuBLAS at M=16

Without tensor cores, scalar fp32 FMA throughput can't keep up. At M=16,
K=4096, N=14336 (MLP up):

- **cuBLAS fp16:** reads 112 MB of fp16 weights, time 138 µs ≈ **812 GB/s**
  — 80% of peak HBM, *bandwidth-bound on fp16 weights via tensor cores*.
- **Our kernel:** reads 28 MB of int4 weights (4× less!), time 239 µs ≈
  **117 GB/s** — well below HBM peak. *Compute-bound on scalar fp32 FMA*.

So our int4 weight savings *should* make us win on bandwidth, and they
would — except the scalar fp32 accumulate path can't process the streamed
int4 weights fast enough to keep the HBM channel saturated. On RTX 4090
(sm_89): fp32 scalar peak ≈ 82 TFLOPs vs fp16 MMA peak ≈ 165 TFLOPs.
Tensor cores are the missing piece.

### What's still open

- **Phase 6 — tensor-core W4A16.** Rewrite the batched-decode kernel with
  `mma.sync` (PTX MMA instruction, 16×8×16 shape on sm_89). This is what
  Marlin / AWQ-V2 / GPTQMarlin all do — and the only path to closing the
  remaining 1.7× gap to cuBLAS at M=16. The Phase 5 kernel makes a
  reasonable scaffold for it: same threading model, same K-split, same
  shmem layout, just swap the inner scalar FMA loop for MMA.
- **`ncu` profile + roofline plot** for the Phase 4 decode step on the
  patched model. Pending GPU clock lock + a quiet system.
- **Calibration-aware quant (GPTQ / AWQ)** to recover the W4A16 accuracy
  hit. Out of scope; flagged for any future "production quality" pass.
- **vLLM integration.** Phase 4 lives entirely on HF transformers; the
  same patches would not drop into vLLM cleanly because vLLM has its
  own PagedAttention + worker model. A vLLM port is a separate effort.

### Phase 5 lesson

**An M-aware kernel without tensor cores is necessary but not sufficient
for batched-decode W4A16 wins.** Phase 5 proves the BLOCK_M amortization
works structurally — the per-block time scales BLOCK_M-rows-of-work, not
M-rows-of-work — but the residual cuBLAS gap is a scalar-vs-tensor-core
question, not a memory-pattern question. The fix is well-known (MMA),
just out of this phase's scope.

## Reference workload reproduction

Same harness as the prior phases:

```bash
# Vanilla baseline (run once; numbers feed into the comparison table)
python scripts/run_phase4_eval.py --step vanilla

# Per-step (each ~30-90 min depending on the step)
python scripts/run_phase4_eval.py --step 4a
python scripts/run_phase4_eval.py --step 4b
python scripts/run_phase4_eval.py --step 4c

# Smoke-test the full pipeline at a small sample (PPL + greedy + 10-sample lm-eval)
python scripts/run_phase4_eval.py --step 4b --limit 50 --skip-tokens-per-sec
```

Per-config JSON outputs land in `docs/results/{e2e_eval,lm_eval}/*.json`.
The greedy reference (10 prompts × 64 new tokens, set by the vanilla run)
is committed at `docs/results/e2e_eval/vanilla_reference_outputs.json`;
delete and re-run vanilla to regenerate it.
