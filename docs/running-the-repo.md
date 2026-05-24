# Running this repo

A guide for reproducing every result on the project. For each phase you'll
see **what to run**, **what numbers to expect**, and **what to look for**
(the underlying lesson).

Cross-references:
- The narrative for Phase 1 (v0 → v5) lives in
  [`01-fused-attention-journey.md`](01-fused-attention-journey.md).
- The narrative for Phase 2 (2a → 2d) lives in
  [`02-kv-cache-compression-journey.md`](02-kv-cache-compression-journey.md).
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

# Should print "decode attention ... 0.71 ms ... 189 GB/s"
python benchmarks/bench_attention.py
```

If both work, the extension is built correctly and `main` is in the
v3 + INT4 KIVI state.

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

`main` is currently the v3 + INT4 KIVI state. After exploring a branch,
return with `git checkout main && python setup.py build_ext --inplace`.

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
| **`v3`**     | `ccdb6df`  | **0.713 ms** | **189 GB/s** | **2.34×** | Vectorized 64-bit KV loads, single-warp block — **on `main`** |
| `v4`         | `f904aae`  | 0.802 ms | 167 GB/s | 2.08×  | Split-K (FlashDecoding) — **regressed, reverted**   |
| `v5`         | `78a28ff`  | 0.760 ms | 177 GB/s | 2.20×  | `cp.async` double-buffer — **regressed, reverted**  |

PyTorch SDPA (FA / cuDNN) lands at **1.36 ms** on this workload. So `v3`
beats the official FlashAttention dispatch by **1.91×**.

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

**`v4` / `v5`** — both *regressed*. Read these for the bandwidth-bound
vs dependency-chain-bound diagnosis:
- `v4` adds more SMs via split-K. Doesn't help — the bottleneck wasn't
  grid utilization.
- `v5` adds explicit `cp.async` pipelining. Doesn't help — the shmem
  hop cost more than the explicit pipeline saved, and nvcc was already
  pipelining loads implicitly.

The diagnostic value here: when bandwidth isn't the ceiling, bandwidth-
targeting optimizations don't help. Phase 2 will hit this lesson again
in 2b.

### Phase 1 microbenchmark

```bash
# Decode attention vs PyTorch eager and SDPA
python benchmarks/bench_attention.py
```

Looks for: PyTorch eager (~3.77 ms), SDPA (~1.36 ms), and the
currently-built custom kernel (whatever branch you're on).

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

## 4. Common gotchas

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

## 5. Where to go from here

- The **journey docs** are where the *why* lives. Each step's
  measurement is in RESULTS.md; the diagnosis is in the journey doc.
  - Phase 1: [`01-fused-attention-journey.md`](01-fused-attention-journey.md)
  - Phase 2: [`02-kv-cache-compression-journey.md`](02-kv-cache-compression-journey.md)
- The **design docs** are forward-looking specs.
  - Phase 1 (decode attention): [`01-fused-attention.md`](01-fused-attention.md)
  - Phase 2 (KV compression): [`02-kv-cache-compression.md`](02-kv-cache-compression.md)
  - Phase 3 (W4A16 GEMM): [`03-quantized-matmul.md`](03-quantized-matmul.md) — not started
- The **TODO list** is the work tracker.
  - [`../TODO.md`](../TODO.md)
- The **benchmarking methodology** locks the reference workload.
  - [`benchmarking-methodology.md`](benchmarking-methodology.md)

**If you only have 10 minutes**:
1. Read the Phase 1 journey doc § "Summary" + "Lessons we want to
   carry forward" — that's the high-density payload.
2. Same for Phase 2 journey doc.
3. `git checkout v3 && pytest tests/test_attention.py` — see the
   correctness gate.
4. `python benchmarks/bench_attention.py` — see the headline number.
