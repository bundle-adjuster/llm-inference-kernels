# Running this repo

A guide for reproducing every result on the project. For each phase you'll
see **what to run**, **what numbers to expect**, and **what to look for**
(the underlying lesson).

Cross-references:
- The narrative for Phase 1 (v0 → v5) lives in
  [`01-fused-attention-journey.md`](01-fused-attention-journey.md).
- The narrative for Phase 2 (2a → 2d) lives in
  [`02-kv-cache-compression-journey.md`](02-kv-cache-compression-journey.md).
- The narrative for Phase 3 (3a → 3c) lives in
  [`03-quantized-matmul-journey.md`](03-quantized-matmul-journey.md).
- The narrative for Phase 4 (e2e integration), Phase 5 (batched-decode
  W4A16) and Phase 6 (tensor-core W4A16) lives in
  [`04-end-to-end-integration-journey.md`](04-end-to-end-integration-journey.md)
  — including the Phase 5/6 addenda.
- Per-step quantitative table: [`results/RESULTS.md`](results/RESULTS.md).
- Reference workload definition (locked): [`benchmarking-methodology.md`](benchmarking-methodology.md).

All numbers below are from RTX 4090 (sm_89). Other GPUs will give
different absolute numbers but the same relative wins/regressions.

---

## 0. One-time setup

```bash
# 1. reproducible conda env (Python 3.11, CUDA 12.4 toolkit)
conda env create -f environment.yml
conda activate llm-inference-kernels

# 2. flash-attn (build takes 5-10 min)
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation

# 3. confirm CUDA visible
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# expect: True 12.4

# 4. environment report (written to docs/results/env-report.md)
bash scripts/detect_env.sh

# 5. build the PyTorch C++/CUDA extension in-place
python setup.py build_ext --inplace
# produces llmik_cuda.cpython-311-x86_64-linux-gnu.so at repo root
```

The model weights for Phase 0 / Phase 2d (Llama 3.1 8B Instruct) need a
gated Hugging Face access token configured (`huggingface-cli login`) on
first use. The download is ~16 GB.

### Smoke test — does it all hang together?

```bash
# Should print "42 passed"
pytest tests/

# Should print "decode attention ... ~0.16 ms ... ~82% of peak HBM"
# (v6 split-K, Phase 8; the retired v3 kernel printed 0.71 ms / 189 GB/s)
python benchmarks/bench_attention.py
```

If both work, the extension is built correctly and `main` is in the
v6 split-K attention + INT4 KIVI + W4A16 + Phase 6 tensor-core state
(every kernel landed through Phase 8 is on the dispatch path;
`decode_attention` dispatches to the v6 FlashDecoding split-K kernel,
with the old kernel preserved as `decode_attention_v3`).

---

## 1. Phase 0 — environment + serving baselines

**What it shows**: vendor production baselines (HF `generate()`, vLLM)
on the reference workload. Everything in Phase 1/2 is measured against
this.

**What to run**:

```bash
# Verify Llama 3.1 8B Instruct loads + generates (~16 GB VRAM)
python scripts/load_llama.py

# End-to-end serving baselines (HF generate vs vLLM)
python benchmarks/bench_e2e.py
```

**What to expect** (locked reference workload: batch=16, prompt=512,
gen=512):

| Stack                  | Latency  | Throughput   |
|------------------------|---------:|-------------:|
| HF `generate()` fp16   | 23.10 s  | 354.6 tok/s  |
| vanilla vLLM 0.6.6     | 11.65 s  | 703.2 tok/s  |

**What to look for**: vLLM is ~2× HF `generate()`. That's the production
gap our Phase 1/2 kernels chip into, one kernel at a time.

---

## 2. Phase 1 — fused decode attention (branches `v0` → `v5`)

Each kernel version lives on its own branch. To check one out:

```bash
git checkout v<N>                # e.g. v3, v4, v5, v1-naive, ...
python setup.py build_ext --inplace   # rebuild the extension
python benchmarks/bench_attention.py  # bench the current kernel
pytest tests/test_attention.py        # confirm correctness
```

`main` is currently the v6 split-K attention + INT4 KIVI state
(`decode_attention` → v6; v3 preserved as `decode_attention_v3`). After
exploring a branch, return with
`git checkout main && python setup.py build_ext --inplace`.

### What to expect, per branch

Reference workload: Llama 3 8B head config (`n_heads=32, n_kv_heads=8,
head_dim=128`), `batch=8, seqlen_kv=4096`, fp16. The numbers below are
the headline latency for `bench_attention.py` (median of 100 timed runs
after 25 warmup).

| Branch       | Commit     | Latency  | KV BW    | vs v0  | What's new                                          |
|--------------|------------|---------:|---------:|-------:|-----------------------------------------------------|
| `v0`         | `46930c2`  | 1.669 ms | 80 GB/s  | 1.00×  | Naive two-pass softmax, shmem score buffer          |
| `v1-naive`   | `46ae1ea`  | 2.078 ms | 65 GB/s  | 0.80×  | Online softmax — **regression** (V load not prefetched) |
| `v1`         | `ad9c57f`  | 1.637 ms | 82 GB/s  | 1.02×  | Same + explicit V prefetch                          |
| `v2`         | `db6ab0b`  | 1.069 ms | 126 GB/s | 1.56×  | Single-sync block reduce (double-buffered shmem)    |
| **`v3`**     | `ccdb6df`  | **0.713 ms** | **189 GB/s** | **2.34×** | Vectorized 64-bit KV loads, single-warp block — retired on `main`, superseded by v6 split-K (Phase 8, see [06](06-attention-splitk-journey.md)) |
| `v4`         | `f904aae`  | 0.802 ms | 167 GB/s | 2.08×  | Split-K (FlashDecoding) — **regressed, reverted**   |
| `v5`         | `78a28ff`  | 0.760 ms | 177 GB/s | 2.20×  | `cp.async` double-buffer — **regressed, reverted**  |

Against a **GQA-native** SDPA baseline
(`F.scaled_dot_product_attention(..., enable_gqa=True)`, **157.3 µs** on
this workload) the single-warp `v3` above actually **loses**: 713.7 µs
vs 157.3 µs — 4.55× slower (**0.22×**). `v3` is occupancy-bound (its
single-warp block fills ~2 of 128 SMs, ~18% of peak HBM), *not*
bandwidth-bound, so the retired "1.36 ms SDPA → 1.91× over
FlashAttention" headline was an artefact of a **handicapped baseline**
(SDPA fed a 4×-expanded GQA KV cache). **Phase 8's `v6`** — FlashDecoding
split-K on 4-warp blocks — is what finally beats fair SDPA: **155.6 µs,
1.01×** (and 4.59× over `v3`), at ~82% of peak HBM. On `main`,
`decode_attention` now dispatches to v6; the old kernel is preserved as
`decode_attention_v3`. See
[`06-attention-splitk-journey.md`](06-attention-splitk-journey.md) (the
fix) and
[`05-baseline-correction-journey.md`](05-baseline-correction-journey.md)
(the correction).

### What to look for, per branch

**`v0`** — read this kernel to understand the naive structure: three
phases (scores → softmax → output) with a shmem score buffer. This is
the CUDA-vs-CUDA baseline; everything else improves on it.

**`v1-naive` vs `v1`** — the most instructive pairing in Phase 1.
- `v1-naive` ran a textbook Milakov-Gimelshein online softmax and
  *regressed* to 2.08 ms (0.80× v0). The first wrong hypothesis was
  "too much redundant `exp` work" — we tested that fix and it made
  things even worse (2.26 ms).
- `v1` (with V-prefetch) hits 1.64 ms. The actual problem in `v1-naive`
  was that the V load sat inside the per-`j` `__syncthreads()` barrier
  — nvcc won't hoist a load above a sync, so V latency couldn't hide
  behind the K reduction. The fix: issue `v_j = V[j, tid]` at the
  *top* of the iteration, alongside `k_j`.

**Lessons** (verify against the journey doc § "v1"):
1. SIMT-parallel ALU work is essentially free.
2. `__syncthreads()` is a load barrier the compiler respects — hoist
   memory loads above syncs manually.

**`v2`** — single-sync block reduce. Look for two changes:
1. All warps redundantly do the final cross-warp reduce
   (no `s_bcast` shmem write/read).
2. `reduce_smem` is double-buffered on `j & 1` (so iter j+1's write to
   the *other* slot doesn't race iter j's read).

The 1.53× speedup over `v1` is bigger than the saved sync alone would
predict — removing the shmem broadcast hop was worth another ~400 µs.
The lesson: **removing a shmem hop can dwarf removing the sync itself**.

**`v3`** — single-warp block + vectorized KV loads. The interesting bet:
trade per-SM occupancy (full → 33%) for vec-load throughput + zero
syncs. It paid off — 1.50× over v2 — because the lost warps were idle
at barriers anyway. **Occupancy is a means, not an end.**

**`v4` / `v5`** — both *regressed at the time* on the single-warp `v3`
base. Read these as history — but note the **original v4 diagnosis was
wrong**:
- `v4` added more SMs via split-K and regressed, so we concluded "the
  bottleneck wasn't grid utilization / bandwidth was the ceiling."
  **Phase 8 overturned this.** `v3` was occupancy-bound (single-warp
  block, ~18% of peak HBM, ~2 of 128 SMs), and split-K rebuilt on
  **4-warp blocks** — the Phase 8 `v6` kernel — is exactly what fixed
  it: v6 reaches ~82% of peak HBM and beats fair SDPA. Grid/occupancy
  *was* the ceiling; split-K done right is the win. See
  [`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).
- `v5` adds explicit `cp.async` pipelining. Doesn't help — the shmem
  hop cost more than the explicit pipeline saved, and nvcc was already
  pipelining loads implicitly. (This diagnosis still stands.)

The dependency-chain-bound vs bandwidth-bound framing still carries into
Phase 2's 2b; only the v4 "bandwidth was the ceiling" conclusion was
retired.

### Phase 1 microbenchmark

```bash
# Decode attention vs PyTorch eager and SDPA
python benchmarks/bench_attention.py
```

Looks for: PyTorch eager (~3.77 ms), SDPA, and the currently-built
custom kernel (whatever branch you're on). The SDPA figure this script
prints (~1.36 ms) is the **handicapped 4×-expanded-GQA** baseline
retired in Phase 7 — fair GQA-native SDPA on this workload is ~157 µs
(measured by `benchmarks/bench_decode_step.py`), which Phase 8's v6
kernel matches/beats (1.01×).

---

## 3. Phase 2 — KV-cache compression (branches `2a` → `2d`)

Each sub-step lives on its own branch:

```bash
git checkout 2<a/b/c/d>
python setup.py build_ext --inplace
```

`main` carries the full Phase 2c kernel (INT4 KIVI) + Phase 2d perplexity
script.

### What to expect, per sub-step

| Branch  | Commit    | What's new                                                                |
|---------|-----------|---------------------------------------------------------------------------|
| `2a`    | `3114309` | PyTorch reference for INT8 / INT4 quantize + tests (no CUDA changes)      |
| `2b`    | `6df94ef` | INT8 per-token CUDA quantize + INT8 attention kernel w/ fused dequant     |
| `2c`    | `cccdced` | KIVI INT4 (per-channel K, per-token V, packed 4-bit) + INT4 attention     |
| `2d`    | `4dc66b8` | WikiText-2 perplexity validation script + both targets met                |

### What to run

```bash
# 2a — reference + correctness only (no extension rebuild needed for 2a)
pytest tests/test_kv_cache.py            # 22 reference tests pass

# 2b / 2c — quantize + attention CUDA path
pytest tests/test_kv_cache.py            # 32 tests (2b) or 38 tests (2c)
python benchmarks/bench_kv_cache.py      # memory + latency table

# 2d — model-level perplexity (Llama 3.1 8B, ~3 min for the full sweep)
python scripts/eval_perplexity.py --n-chunks 64
```

### What to expect

On the reference workload (`batch=8, seqlen_kv=4096`, Llama 3 8B heads):

| Variant                              | Memory      | Latency   | Δppl vs fp16 | What changed                                       |
|--------------------------------------|------------:|----------:|-------------:|-----------------------------------------------------|
| fp16 KV (Phase 1 v3, baseline)       | 128.00 MiB  | 0.713 ms  | 0            | Reference for everything below                      |
| INT8 per-token (2b)                  | 65.00 MiB   | 0.713 ms  | +0.0008      | Half the bytes; **tied latency** with v3            |
| INT4 KIVI (2c)                       | **34.50 MiB** | **0.554 ms** | **+0.196** | Quarter the bytes; **1.29× faster than v3**         |
| INT4 per-token K, V (2d comparator)  | n/a         | n/a       | +0.462       | KIVI's "see what naive INT4 looks like"             |

`bench_kv_cache.py` prints the memory + latency table directly.
`eval_perplexity.py` prints the perplexity table (fp16 baseline 7.055,
INT8 +0.0008, INT4-per-token +0.462, INT4 KIVI +0.196).

### What to look for, per sub-step

**`2a`** — read `reference/kv_cache_ref.py` to understand the
quantization conventions. Two axis modes: per-token (V in KIVI) and
per-channel groupwise (K in KIVI, with `group_size=32` along seqlen).
The 22 tests document the round-trip tolerance precisely (1.0/qmax of
the per-axis max).

**`2b`** — INT8 path. The interesting design choice is the
**scale-folding optimization** in the attention kernel:
```
partial = q · k_int                      // 4 muls per thread per iter
partial = warp_reduce_sum(partial)
s_j     = partial · k_scale · softmax_scale     // ONE mul folded in
p_j_v   = p_j · v_scale                  // fold V scale into FMA coeff
o_v[d]  = o_v[d] · alpha + p_j_v · v_int[d]
```
Saves 8 multiplies per iter vs naive dequant-everything (sound because
the dot product is linear in a scalar k_scale).

But — **halving KV bytes did not move latency**. Read the journey doc
section "2b — Surprise: latency tied despite halving KV bytes" for the
diagnosis. Headline: the kernel is dependency-chain-bound (warp_reduce
→ softmax → FMA), not bandwidth-bound. Same lesson as Phase 1's v4/v5.

**`2c`** — INT4 KIVI. The interesting design choice is the
**per-group K-scale pre-fold**:
```cuda
for g in 0..n_groups:
    q_scaled[d] = q_v[d] * k_scale[g, d]    // 4 muls once per group
    for t in g·group_size .. (g+1)·group_size:
        ... inner loop has NO K scale work ...
```
K scales load once per 32 inner iters. Combined with the smaller K_q
and V_q loads (`LDG.E.16` instead of v3's `LDG.E.64`), the inner-loop
dependency chain gets meaningfully shorter — which is why 2c moves the
needle (1.29× over v3) where 2b tied. **The chain itself got lighter.**

**`2d`** — perplexity validation. Read `scripts/eval_perplexity.py` for
the patching mechanism: rebind `F.scaled_dot_product_attention` to
round-trip K, V through the PyTorch quantization reference before
delegating to the real sdpa. `LlamaSdpaAttention.forward` looks up
the symbol on the F module at call time, so the rebind intercepts every
layer's attention without touching HF's RoPE/GQA plumbing.

What to look at in the output:
- INT8 Δppl is essentially zero (0.0008) — INT8 KV is a no-brainer drop-in.
- INT4 KIVI Δppl is 0.196 — clears the < 0.5 target.
- **Naive INT4 (per-token K) Δppl is 0.462** — KIVI is **2.36× better**
  at the same bit depth. The K-side outliers really do matter.

---

## 4. Phase 3 — W4A16 quantized matmul (branches `3a` → `3c`)

Each sub-step lives on its own branch:

```bash
git checkout 3<a/b/c>
python setup.py build_ext --inplace
```

`main` carries the full Phase 3c decode-optimized kernel.

### What to expect, per sub-step

| Branch  | Commit    | What's new                                                                |
|---------|-----------|---------------------------------------------------------------------------|
| `3a`    | `b95872f` | PyTorch reference: W4A16 quantize+dequant+matmul + 17 tests               |
| `3b`    | `85f8409` | Naive CUDA W4A16 GEMM — one warp per output tile of 32 cols, outer-M loop |
| `3c`    | `1bbae49` | Decode-optimized: 4-warp blocks, K split across warps, act in shmem       |

### What to run

```bash
# 3a — reference + correctness only (no extension build needed)
pytest tests/test_quant.py            # 17 tests (3a) or 24 (3b/3c)

# 3b / 3c — CUDA W4A16 GEMM
pytest tests/test_quant.py            # 24 tests pass on both branches
python benchmarks/bench_w4a16.py      # M-sweep × 3 Llama shapes vs cuBLAS
```

### What to expect

On Llama 3 8B layer shapes at M=1 (decode), `bench_w4a16.py` prints:

| Shape (K, N)         | fp16 cuBLAS | 3b naive   | 3c decode  |
|----------------------|------------:|-----------:|-----------:|
| 4096 × 4096 (attn)   | 0.047 ms    | 0.088 ms (loss) | **0.016 ms · 2.88×** |
| 4096 × 14336 (MLP up)| 0.134 ms    | 0.084 ms (1.59×) | **0.019 ms · 6.97×** |
| 14336 × 4096 (MLP dn)| 0.133 ms    | 0.284 ms (loss) | **0.045 ms · 2.96×** |

### What to look for, per sub-step

**`3a`** — read `reference/quant_matmul_ref.py`. Symmetric INT4
per-channel groupwise; matches Phase 2c's KIVI K-side math but applied
to a `[K, N]` weight matrix. The Python `pack_int4_along_k` helper
builds the `[K/8, N] int32` packed layout the CUDA kernel reads.

**`3b`** — read `kernels/quant/quant_matmul.cu`. The naive kernel
already hit Phase 3 *Threshold* (M=1, K=4096, N=14336: 1.59× over
cuBLAS). The interesting bits:
- The shift-trick sign-extend: `(int32)(w_packed << (28 - i*4)) >> 28`
  unpacks one nibble from a packed uint32. Arithmetic right shift in
  PTX sign-extends; works in 4 cycles per nibble.
- Coalescing: at iter `k_pack`, the 32 lanes load
  `weight_packed[k_pack, n_base..n_base+31]` — 32 contiguous int32 =
  one 128-byte warp-wide coalesced load.
- The two M=1 *losses* (attn-square and mlp-down) tell us what 3c needs
  to address.

**`3c`** — same file, separate `w4a16_gemm_decode_kernel` function.
The structural changes:
- 4-warp block (128 threads) per output tile of 32 columns. Each
  warp's 32 lanes own the same 32 columns; K is split 4-way across
  the warps.
- After the K loop, a tiny `partials_smem[WARPS × BLOCK_N]` reduction
  sums the 4 per-warp partials per column.
- `act` loaded into shmem cooperatively at kernel start (8 KiB for
  K=4096, 28 KiB for K=14336).
- Launcher dispatches: `M == 1` → decode kernel; `M > 1` → 3b naive
  fallback.

Read the 3b → 3c improvement story in the Phase 3 journey doc —
the win factorises cleanly across the three shapes (5.4× / 4.4× / 6.3×
improvement over 3b) because the two changes (K-split + shmem) attack
two different bottlenecks (sequential K work; SM under-occupation at
small N) that affect the three shapes differently.

---

## 5. Phase 4 — End-to-end integration (branches `phase4-eval-prep` → `phase4-wrapup`)

Phase 4 takes the kernels from Phases 1–3 and plugs them into the actual
Llama 3.1 8B Instruct model via HF monkeypatch, one kernel at a time,
with a full accuracy/latency/memory eval at each step. **This is where
the kernel-level wins meet the real workload and the tradeoffs become
visible.**

Each sub-step has its own branch; `main` carries all of them merged.

```bash
git checkout phase4-<step>
python setup.py build_ext --inplace
```

### What to expect, per sub-step

Reference workload: Llama 3.1 8B Instruct, batch=16, prompt=512, gen=512.
Baseline (vanilla HF): MMLU 68.32%, HellaSwag 79.51%, ARC-C 60.84%,
PPL 7.055, 335.8 tok/s, 18.50 GB peak VRAM.

| Branch                | Commit    | tok/s  | Peak VRAM | MMLU    | PPL   | What's new                                         |
|-----------------------|-----------|-------:|----------:|--------:|------:|----------------------------------------------------|
| `phase4-eval-prep`    | `1c56d82` | 335.8  | 18.50 GB  | 68.32%  | 7.055 | Eval scaffolding + vanilla baseline locked         |
| `phase4-attention`    | `d6d8c88` | 344.1  | 18.57 GB  | 68.32%  | 7.055 | F.sdpa rebind → Phase 1 v3 decode_attention        |
| `phase4-kv-int4`      | `0836a77` | **521.7** | 18.41 GB  | 67.29% | 7.256 | Int4KIVICache (HF Cache subclass) + INT4 attention |
| `phase4-w4a16` (M=1 only) | `8882880` | 40.9  | **9.05 GB** | 62.40% | 8.087 | QuantizedLinear (W4A16 weights) — regressed at B=16 |
| `phase4-wrapup`       | `7da3de4` | (docs) | —         | —       | —     | Journey doc + README headlines + Phase 5/6 stubs   |

The 4c row above shows the *historical* number on the `phase4-w4a16`
branch — it falls back to the v0 naive kernel at M=batch=16, which
re-reads weights 16× per output row. Phase 5 + Phase 6 fix that;
re-running 4c with the current `main` gives 198.7 tok/s.

### What to run

```bash
# Run a single phase 4 step end-to-end (loads model once, applies the
# step's patches, runs e2e + lm-eval). Each step ~30-60 min.
python scripts/run_phase4_eval.py --step vanilla   # baseline (3 min for e2e, ~40 min for lm-eval)
python scripts/run_phase4_eval.py --step 4a        # bit-identical accuracy + tok/s
python scripts/run_phase4_eval.py --step 4b        # INT4 KIVI KV cache: ~1pp MMLU, 1.55x tok/s
python scripts/run_phase4_eval.py --step 4c        # W4A16: -51% peak VRAM, -5.9pp MMLU

# Smoke test a step (PPL + greedy match only — ~3 min)
python scripts/run_phase4_eval.py --step 4b --skip-lm-eval --skip-tokens-per-sec
```

Outputs go to `docs/results/{lm_eval,e2e_eval}/<step_name>.json`. The
vanilla greedy reference (10 prompts × 64 new tokens) lives at
`docs/results/e2e_eval/vanilla_reference_outputs.json`; 4a/4b/4c
compare against it for the greedy-match metric.

### What to look for, per sub-step

**`phase4-eval-prep`** — read `scripts/run_lm_eval.py` +
`scripts/run_e2e_eval.py`. The interesting design choice is that both
accept a *pre-loaded* HF model (not just a `pretrained=<id>` string),
so the 4a/4b/4c orchestrator can pass a patched model in. lm-eval
defaults to creating its own; you have to use
`HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")` to
inject yours.

**`phase4-attention`** — F.sdpa rebind in `integration/attention_patch.py`.
The interesting design choices:
- `q_len == 1` (decode) dispatches to our kernel; `q_len > 1` (prefill)
  falls through to original SDPA.
- HF's `repeat_kv` is undone via `K[:, ::n_rep].contiguous()` — without
  this, our kernel reads 4× the KV bandwidth on Llama 3 8B and loses
  its whole reason for existing.
- Accuracy is **bit-identical** to vanilla on every prefill-based metric
  + `greedy_match=1.0` confirms decode is bit-perfect across 640 tokens.

The +2.5% tok/s is the *interesting* result: attention is only ~4% of
decode time at this workload (batch=16, avg kv_len≈768). This step ran
the `v3` kernel, whose "1.91× microbench" was against a handicapped
(4×-expanded GQA) SDPA baseline — against fair GQA-native SDPA `v3`
actually *lost* (0.22×), so there was never an attention win to amortize
here. Amdahl still sets the ceiling: the e2e win is kernel-speedup ×
fraction-of-time-on-that-kernel, and the kernel that actually clears
1.0× on fair SDPA is **Phase 8's v6** (now the dispatched
`decode_attention`). See
[`06-attention-splitk-journey.md`](06-attention-splitk-journey.md).

**`phase4-kv-int4`** — read `integration/kv_int4_cache.py` (the HF Cache
subclass) + the `patched_int4_decode_attention` context in
`integration/attention_patch.py`. The design choices:
- The cache stores packed INT4 + scales; an fp16 *residual buffer* holds
  recent tokens that don't yet fill a `group_size=32` group. Decode
  appends 1 token at a time; once the residual hits 32, that chunk is
  quantized into the packed storage.
- Two return paths from `update()`: `update()` returns dequantized fp16
  for SDPA compat (used at prefill); `append_only()` skips the fp16
  materialize and is used by the int4 decode fast path.
- Decode replaces `LlamaSdpaAttention.forward` to call our
  `decode_attention_int4` directly on the packed tensors. lm-eval
  prefill uses a separate F.sdpa rebind that quantize-and-dequantizes —
  same KIVI math, same noise.

**The PPL delta of +0.20 matches Phase 2c's kernel-level +0.196** —
direct validation that the integration faithfully reproduces the
kernel's noise pattern. The 1.55× tok/s is bigger than Phase 2c's 1.29×
microbench because the forward replacement *also* skips `repeat_kv` and
SDPA dispatch overhead.

**`phase4-w4a16`** — read `integration/w4a16_patch.py`. The
`QuantizedLinear` class replaces 224 `nn.Linear` modules in place
(7 per layer × 32 layers). Weight VRAM drops from 16.06 GB to 5.70 GB
immediately after the patch.

The interesting result: at the locked workload (batch=16), 4c
*regressed* on tok/s. The Phase 3 W4A16 kernel was designed for M=1
(single-stream decode); at M=batch=16, the launcher falls back to v0
naive, which re-reads weights 16× per output row. **This is exactly the
lesson Phase 3 had flagged**: "batched-decode M>1 is integration
concern (Phase 4)" — 4c is where it surfaces.

Phase 5 and Phase 6 (next two sections) close most of that gap; the
4c row in RESULTS.md reflects the *current* (Phase 6) dispatch.

---

## 6. Phase 5 — Batched-decode W4A16 kernel (branch `phase5-batched-w4a16`)

Resolves the Phase 4c regression at batch=16 by adding
`w4a16_gemm_batched_decode_kernel`. Same K-split-across-warps pattern
as the Phase 3c decode kernel, but each thread accumulates a length-
`BLOCK_M=16` vector of fp32 partials, and each warp has its own
`[BLOCK_M, group_size]` activation tile in shmem so the int4 weight
stream is amortized across all M rows.

The launcher dispatch becomes:

```
M == 1            → v1 decode (Phase 3c)
M in [2, 16]      → v2 batched-decode (Phase 5)
M  > 16           → v0 naive (Phase 3b)
```

```bash
git checkout phase5-batched-w4a16
python setup.py build_ext --inplace

# Microbench at the three Llama 3 8B shapes, M-sweep:
python benchmarks/bench_w4a16.py

# Re-run Phase 4c end-to-end with the new kernel:
python scripts/run_phase4_eval.py --step 4c --skip-lm-eval
```

### What to expect

| Phase 4c at batch=16                | tok/s     | Notes                                          |
|-------------------------------------|----------:|------------------------------------------------|
| M=1 kernel only (`phase4-w4a16`)    | 40.9      | naive M>1 fallback — 16× weight bandwidth      |
| + Phase 5 batched-decode kernel     | **199.9** | **4.9× recovery; 0.60× vs vanilla HF (335.8)** |

Kernel-level at M ∈ [2, 16]: near-flat ~200 µs/call across all shapes
(the work is BLOCK_M-sized regardless of actual M ≤ BLOCK_M — exactly
the amortization we wanted). Still ~1.7× behind cuBLAS at M=16 because
cuBLAS uses tensor cores; ours used scalar fp32 FMA. Closing that is
Phase 6.

### What to look for

Read `kernels/quant/quant_matmul.cu::w4a16_gemm_batched_decode_kernel`.
Two interesting differences vs Phase 3c v1:

1. **Per-warp activation tile in shmem.** v1 cached the *full* K-wide
   activation in one shared `act_smem[K]`; that worked because M=1 so
   the tile was small. v2 has BLOCK_M=16 rows, which would be 16×
   bigger — too big for shared memory at Llama 3 K=14336. The fix:
   each warp gets its own slice of K, so the per-warp tile only spans
   one K group (BLOCK_M × group_size = 16 × 128 fp16 = 4 KB per warp,
   16 KB across 4 warps).
2. **Per-thread vector accumulator.** Each thread holds 16 fp32
   accumulators (one per M row of its column). The inner FMA loop
   reads one act value (broadcast across the warp) and multiplies it
   against the column's weight nibble — 16 FMAs per nibble per thread.

The lesson: the M-amortization works structurally (kernel time is flat
across M ∈ [2, 16]), but **scalar fp32 FMA throughput can't keep up
with the int4 weight bandwidth** at this M. Per-Linear time is
~200 µs, compute-bound, while cuBLAS is ~140 µs at the same shape,
*bandwidth-bound on fp16 weights*. We read 4× fewer bytes than cuBLAS
but can't process them fast enough. Tensor cores are what closes that.

---

## 7. Phase 6 — Tensor-core W4A16 kernel (branch `phase6-tensorcore-w4a16`)

Drops `mma.sync` (via the wmma C++ API, `m16n16k16` fp16→fp32) into the
Phase 5 batched-decode kernel's inner accumulate loop. Same M=16
batched-decode shape, same threading model, same dequant path — only
the FMA path swaps.

The launcher dispatch becomes:

```
M == 1            → v1 decode (Phase 3c)
M in [2, 16]      → v3 tensor-core (Phase 6)   -- v2 kept in file as reference
M  > 16           → v0 naive (Phase 3b)
```

```bash
git checkout phase6-tensorcore-w4a16
python setup.py build_ext --inplace

# Microbench
python benchmarks/bench_w4a16.py

# Re-run Phase 4c
python scripts/run_phase4_eval.py --step 4c --skip-lm-eval
```

### What to expect

**Kernel-level vs Phase 5 v2 at M=16** (Llama 3 8B shapes):

| Shape (K, N)             | v2 scalar | v3 tensor-core | improvement |
|--------------------------|----------:|---------------:|------------:|
| 4096 × 4096 (QKV/O)      | 211 µs    | **152 µs**     | **1.39×**   |
| 4096 × 14336 (MLP up/gate)| 239 µs   | **167 µs**     | **1.43×**   |
| 14336 × 4096 (MLP down)  | 686 µs    | **514 µs**     | **1.33×**   |

vs cuBLAS at M=16: 0.37× / 0.81× / 0.35× respectively. Closer than v2
on every shape; closest to parity on the MLP up/gate (the headline
shape for decode throughput).

**E2E Phase 4c at batch=16**: 198.7 tok/s — *essentially unchanged*
from Phase 5's 199.9. **The microbench win doesn't show up end-to-end.**

### What to look for

Read `kernels/quant/quant_matmul.cu::w4a16_gemm_tc_kernel`. Compared
to Phase 5 v2:

- `BLOCK_N` widens from 32 (one warp's lanes) to 64 (4 warps × 16
  cols/warp). No K-split-across-warps — each warp processes the full
  K independently across its own col tile.
- The inner accumulate loop iterates `group_size / WMMA_K = 128 / 16
  = 8` MMA calls per K group, consuming the dequantized weight tile
  via `load_matrix_sync(b_frag, ...)`.
- Output is staged through shmem because rows `[M, 16)` are
  zero-padded; the cooperative write skips them.

**Why the e2e didn't move:** a back-to-back microbench of 7 same-shape
calls shows 173 µs/call vs 153 µs single-call — only 13% inter-call
overhead, so launch saturation isn't the cause. The likely bottleneck
is **host-side Python/PyTorch dispatch overhead per Linear call**:
~5-30 µs × 224 Linears × 512 decode steps ≈ 5-30 s of host work
across a ~41 s run. At Phase 5's kernel time the GPU was the
bottleneck; at Phase 6's the host is. The kernel speedup is real but
masked by host overhead.

**The lesson:** a microbench win can be invisible end-to-end if a
*different* bottleneck moves into the gap. Closing the e2e gap now
needs host-side optimization (CUDA graphs / `torch.compile` to capture
one decode step as a single GPU command), not more kernel work. That's
queued as Phase 7 in TODO.md.

---

## 8. Common gotchas

**After `git checkout <branch>`, always rebuild**:
```bash
python setup.py build_ext --inplace
```
The `.so` binary in the repo root is for whichever branch built last.
If you check out `v0` and run the bench without rebuilding, you're
benchmarking whatever was previously built (probably v3) — not v0.

**Always activate the conda env in fresh shells**:
```bash
conda activate llm-inference-kernels
```
Forgetting this is the #1 cause of `ModuleNotFoundError: No module named
'llmik_cuda'` — the build went into `llm-inference-kernels`'s
site-packages, and the current shell is using a different python.

**Run from the repo root**. `setup.py build_ext --inplace` puts the
`.so` at the repo root, and the benchmarks use relative imports
(`benchmarks/`, `reference/`).

**For perplexity eval, expect the first chunk to be slow.** The first
forward pass triggers CUDA kernel JIT / cuDNN autotuning. The reported
tokens/sec excludes warmup but the wall-clock includes it.

**`pytest tests/test_attention.py` requires the extension built**.
The kv_cache pure-reference tests don't (they're pytorch-only), but
the attention tests use `import llmik_cuda`.

**vLLM 0.6.6 is pinned in environment.lock.yml**. Newer vLLM versions
have different API surface; if you upgrade, the `bench_e2e.py` script
will need touching.

---

## 9. Where to go from here

- The **journey docs** are where the *why* lives. Each step's
  measurement is in RESULTS.md; the diagnosis is in the journey doc.
  - Phase 1: [`01-fused-attention-journey.md`](01-fused-attention-journey.md)
  - Phase 2: [`02-kv-cache-compression-journey.md`](02-kv-cache-compression-journey.md)
  - Phase 3: [`03-quantized-matmul-journey.md`](03-quantized-matmul-journey.md)
  - Phase 4 (+ Phase 5/6 addenda): [`04-end-to-end-integration-journey.md`](04-end-to-end-integration-journey.md)
- The **design docs** are forward-looking specs.
  - Phase 1 (decode attention): [`01-fused-attention.md`](01-fused-attention.md)
  - Phase 2 (KV compression): [`02-kv-cache-compression.md`](02-kv-cache-compression.md)
  - Phase 3 (W4A16 GEMM): [`03-quantized-matmul.md`](03-quantized-matmul.md)
- The **TODO list** is the work tracker.
  - [`../TODO.md`](../TODO.md) — Phase 7 (CUDA graphs to hide host
    overhead) is the next queued item.
- The **benchmarking methodology** locks the reference workload.
  - [`benchmarking-methodology.md`](benchmarking-methodology.md)

**If you only have 10 minutes**:
1. Read the Phase 4 journey doc § "Summary" + "Lessons we want to
   carry forward" — the highest-density view of where every kernel
   actually ends up on the real model.
2. `git checkout main && pytest tests/` — see the correctness gate
   across every kernel landed.
3. `python benchmarks/bench_attention.py` and
   `python benchmarks/bench_w4a16.py` — the headline kernel-level numbers.

**If you have an afternoon**:
1. Read the Phase 1 journey doc — the diagnostic-framework lessons
   (dependency-chain-bound vs bandwidth-bound; v1/v4/v5 each illustrate
   one) carry into Phase 2 and Phase 5.
2. Read the Phase 4 journey doc, including the Phase 5 and Phase 6
   addenda at the bottom — the "kernel-level win doesn't always land
   end-to-end" story is unique to Phase 4-6 and is the most surprising
   result in the project.
3. `python scripts/run_phase4_eval.py --step 4b --skip-lm-eval` —
   ~3 minutes; see the KIVI accuracy hit on real Llama 3.1 8B.
