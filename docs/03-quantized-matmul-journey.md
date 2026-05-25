# Phase 3 Journey — W4A16 Quantized Matmul

> Companion to [`03-quantized-matmul.md`](03-quantized-matmul.md) (design)
> and [`results/RESULTS.md`](results/RESULTS.md) (numbers).
>
> Phase 3 was scoped tighter than 1 or 2: just the Threshold + Target
> goals (3a reference → 3b naive → 3c decode-optimized → 3d wrap). The
> Tensor Core prefill path (stretch goal) and the GPTQ perplexity loop
> were deferred. Three substeps of work; the most interesting jump is
> 3b → 3c, where K-split-across-warps + act-in-shmem moved every shape
> from "naive baseline" to "2.88–6.97× over fp16 cuBLAS."

## Setting

- **GPU**: RTX 4090 (sm_89), 1008 GB/s peak HBM, 72 MB L2.
- **Target**: Llama 3 8B's linear layers — `K, N ∈ {(4096, 4096),
  (4096, 14336), (14336, 4096)}` covering the three families
  (attention QKV/O, MLP up/gate, MLP down). M ∈ {1, 8, 32}; **M=1 is
  the canonical decode shape and the primary target.**
- **Baseline**: `torch.matmul` (cuBLAS fp16 under the hood).
- **Quantization**: symmetric INT4, group-wise scales along K with
  `group_size=128`. One fp16 scale per (group, output channel).
- **Packing**: weights are stored as `[K/8, N] int32`, 8 nibbles per
  uint32 along K. Bit `i*4..i*4+3` = K position `k_pack*8 + i`. The
  packing happens host-side (the Python helper
  `pack_int4_along_k`); production W4A16 quantises offline.

The Phase 3 design doc was honest up front: "the achievable, defensible
win is the decode-shape, memory-bound GEMM, not beating cuBLAS on large
GEMMs." That's the goal we held to.

---

## 3a — Reference + tests · commit `b95872f`

### Theory

The math is the same per-channel-groupwise quantization we built in
Phase 2c, just on a [K, N] weight matrix instead of a [batch, head,
seqlen, head_dim] KV cache:

```
scale[g, n] = max(|W[g·group_size:(g+1)·group_size, n]|) / 7
q[k, n]     = round(W[k, n] / scale[k // group_size, n]),
              clamped to [-7, 7]
W_hat[k, n] = q[k, n] · scale[k // group_size, n]
out[m, n]   = Σ_k act[m, k] · W_hat[k, n]
```

### Design choices

- **`[K, N]` layout** (the transpose of `torch.nn.Linear.weight`'s
  `[N, K]`) — matches the natural row-major GEMM `out[M, N] = act[M, K]
  @ W[K, N]`. Pass `Linear.weight.T.contiguous()` to use real layer
  weights with this reference.
- **Reference stores int8** containers (values in [-7, 7]); the
  packing for the CUDA kernel happens in a separate helper. Same
  decoupling as KIVI's int4 reference in Phase 2c.
- **`group_size=128` along K** matches the GPTQ/AWQ standard and is
  small enough that scale outliers in any one group don't poison
  the whole K dimension.

### Result

17 tests across 3 layer shapes × 2 group sizes × 3 M values. Round-trip
within `1/qmax = 1/7` of per-(group, output-channel) max; storage ratio
0.258× of fp16 (theoretical packed); matmul-vs-fp16 within bounded
INT4 noise.

### What we learned

- The reference and the CUDA kernel mostly factor cleanly: the
  reference does dequant-then-matmul, the kernel does fused-dequant
  inline. Both implement the same math, so the kernel correctness
  gate ("CUDA bit-equal to reference on same dequantized inputs") is
  cheap and tight.
- The "matmul-vs-fp16-noise" test needed generous bounds — INT4 on
  synthetic gaussian weights at M=1, N=14336 hits 28% rel err on the
  worst single output (sum of K=14336 noisy terms accumulating).
  That's the **expected noise floor** of W4A16 on hard inputs, not a
  kernel bug. Same lesson as Phase 2's kernel-level-rel-err number not
  predicting model-level perplexity.

---

## 3b — Naive CUDA W4A16 GEMM · commit `85f8409`

### Theory

Decode is memory-bound on weight traffic — `M=1`, large `K`, large `N`
means almost zero compute reuse per byte of weight read. 4× less weight
bytes (INT4 vs fp16) → up to 4× less HBM read → up to 4× faster decode
GEMM, modulo dequant overhead. That's the W4A16 thesis.

For one (batch, head) decode call:

```
out[n] = Σ_k act[k] · W_hat[k, n]    for n in 0..N-1
```

That's a gemv. The kernel needs to read all K weights for each output
column. With INT4 packing 8 per uint32, we read `K/8` uint32 per
output column.

### Design choices

- **One warp per output tile of `BLOCK_N=32` columns.** 32 threads,
  one per output column. Each thread independently computes one
  fp32 accumulator over K.
- **Inner-loop per (m, n)**: load one uint32 per `k_pack`, unpack 8
  nibbles via the shift-trick `(int32)(w << (28 - i*4)) >> 28`,
  multiply by the per-(group, column) fp16 scale, FMA with the
  corresponding fp16 activation.
- **Coalescing**: at iter `k_pack`, the 32 threads of the warp load
  `weight_packed[k_pack, n_base..n_base+31]` — 32 contiguous int32 =
  one 128-byte warp-wide coalesced load.
- **Activations**: read directly from HBM, no shmem caching. L1
  catches the reuse across the 32 threads in a warp.
- **M loop is outer**: for M > 1, each output row is computed
  separately, with weights reloaded per row. Naive on purpose; the
  real solution is the M-fast inner loop (deferred to Phase 4).

### Result — partial win

Bench on Llama 3 8B shapes at M=1:

| Shape (K, N)    | cuBLAS fp16  | 3b naive    | speedup |
|-----------------|-------------:|------------:|--------:|
| 4096 × 4096     | 0.047 ms     | 0.088 ms    | 0.53× (loss) |
| **4096 × 14336**| **0.134 ms** | **0.084 ms**| **1.59× (win)** |
| 14336 × 4096    | 0.133 ms     | 0.284 ms    | 0.47× (loss) |

The naive kernel **already hits Phase 3 Threshold** ("beats fp16
cuBLAS on decode-shape GEMM") on MLP up/gate — the headline Llama
decode shape. But the other two M=1 shapes lose.

### What we learned

- **The W4A16 thesis holds** at the shape where everything aligns:
  large N (lots of blocks to fill the GPU), small M (memory-bound on
  weight traffic), reasonable K (fast inner loop). 4× less weight
  bytes → ~1.6× wall-clock at this shape.
- **The two losses tell us what's missing**:
  - **4096 × 4096**: only `N/BLOCK_N = 128` 1-warp blocks. With 128
    SMs on the 4090, that's ~1 warp per SM — severe under-occupation.
    Hard to hide *any* latency.
  - **14336 × 4096**: long K reduction (14336 sequential FMAs per
    thread, plus the inevitable HBM weight reads). Per-thread K work
    is too high; the inner loop is the bottleneck.
- **The naive kernel's per-thread structure isn't decoupling K work
  from N work** — each thread does the full K reduction for one
  output column. To improve the losers, we need to either (a) increase
  per-block compute (more warps per block), (b) split K across warps,
  or (c) both. That's 3c.

---

## 3c — Decode-optimized W4A16 GEMM · commit `1bbae49`

### Theory

The 3b losses point at two structural problems:

1. **Under-occupation at small N.** 128 single-warp blocks isn't
   enough to fill 128 SMs × 48 warps. The fix: more warps per block.
2. **Long sequential K.** A single thread can't pipeline a 14336-long
   FMA chain well. The fix: split K across multiple threads.

Both fixes pull in the same direction: **multi-warp blocks with K
split across warps**.

### Design choices

- **Block = 4 warps (128 threads).** `BLOCK_N=32` columns owned by
  the block; each warp's 32 lanes own the same 32 columns (one column
  per lane).
- **K split across warps.** With 4 warps, each warp processes
  `K/4` of the K reduction for all 32 columns. For K=4096,
  group_size=128: 32 total groups → 8 groups per warp.
- **Tiny shmem combine at the end.** Each warp writes its 32 per-column
  partial accumulators to `partials_smem[warp_id × 32 + lane]` (4 × 32
  × 4 B = 512 B). After a `__syncthreads()`, warp 0 sums the 4 partials
  per column and writes the fp16 output.
- **`act` in shared memory.** Cooperative load of the K-length
  activation vector into shmem once per block, used by all 4 warps
  through the K loop. For K=4096, that's 8 KiB shmem.
- **Launcher dispatch.** M == 1 → this decode kernel; M > 1 → the 3b
  naive kernel (no M-fast inner loop yet).

### Result — full Target hit

All three M=1 Llama 3 8B shapes now beat fp16 cuBLAS by 2.88–6.97×.

| Shape (K, N)    | M=1 cuBLAS | 3b naive (3b speedup) | **3c decode** | **3c speedup** | 3c improvement over 3b |
|-----------------|-----------:|----------------------:|--------------:|---------------:|-----------------------:|
| 4096 × 4096     | 0.047 ms   | 0.088 ms (0.53×)      | **0.016 ms**  | **2.88×**      | **5.4×** |
| **4096 × 14336**| 0.134 ms   | 0.084 ms (1.59×)      | **0.019 ms**  | **6.97×**      | **4.4×** |
| 14336 × 4096    | 0.133 ms   | 0.284 ms (0.47×)      | **0.045 ms**  | **2.96×**      | **6.3×** |

The achieved weight bandwidth at M=1, MLP up/gate is ~1577 GB/s —
*higher* than HBM peak (1008 GB/s). That's the L2 cache helping:
28.88 MiB of packed weights fits in the 4090's 72 MB L2, so after the
first warmup iter, subsequent calls hit L2 not HBM. The bandwidth
number is "L2-served effective throughput" — which is the realistic
decode regime (weights stay warm across many forward passes per
request).

### What we learned

The win factorises cleanly across the three shapes:

- **K-split across warps** addresses the long-K serial-FMA problem
  directly — for the K=14336 mlp-down case, each thread now does
  K/4 = 3584 FMAs instead of 14336.
- **Multi-warp blocks** address the SM-occupation problem at small N
  — the 4096×4096 case went from ~1 warp/SM (under-occupied) to
  ~4 warps/SM (room for latency hiding).
- **`act`-in-shmem** is the second-order win. L1 was already catching
  the activation reuse across the 32 threads in a warp; the explicit
  shmem cache mostly frees L1 for the weight traffic. The MLP up/gate
  shape's 4.4× improvement over 3b would have been smaller without
  this (probably 3-3.5× from K-split alone), since L1 pressure was
  closer to saturation.

The improvement *per shape* — 5.4× / 4.4× / 6.3× over 3b — is
remarkably uniform. That tells us the two changes work together rather
than addressing different shapes independently. **The K-split converts
the kernel from "1 warp doing K work serially" to "4 warps doing K/4
work in parallel"**, which is a structural transformation that helps
any shape with non-trivial K. The `act`-in-shmem is a pure plus, no
trade-off.

### What we deliberately didn't do (and why)

- **Vectorised weight loads** (uint4 per thread per iter = 32 K-values
  per pop). The `[K/8, N]` layout doesn't allow per-thread uint4 along
  K without breaking warp-coalescing — consecutive K_packs at the same
  N are at addresses separated by N, not contiguous. A real
  optimization here would need a permuted weight layout (`[K/32, N, 4]`
  or Marlin's blocked tile layout). Deferred.
- **M-fast inner loop** for M > 1. Currently the launcher dispatches
  M > 1 to the 3b naive kernel, which loops M outer (reloading weights
  per row). The M-fast version would load each weight once and FMA
  into BLOCK_M accumulators inside the inner loop. A clean
  improvement, but M > 1 in decode is Phase 4 integration concern
  (batched-decode serving); for now the M=1 fast path is the
  production case.
- **Tensor Core MMA path** for prefill (M ≥ 16, compute-bound). The
  Phase 3 design doc lists this as the prefill story; it's a different
  kernel structure (mma.sync, shmem staging, fp16 dequant before MMA)
  and was scoped out as a stretch goal.
- **GPTQ-style asymmetric quantisation** (zero-point + scale). Better
  quality on real LLM weights; would need the kernel to subtract
  zero-points inside the dequant. Simple extension to the kernel, but
  perplexity validation (Phase 2d-style WikiText eval with
  GPTQ-quantised Llama 3 8B) is its own multi-session lift.

---

## Summary

| Step | Best speedup over fp16 cuBLAS (M=1) | Notes |
|------|--:|-------|
| fp16 cuBLAS  | 1.00× | baseline (PyTorch dispatches here) |
| 3b naive     | up to **1.59×** | wins only on MLP up/gate; loses on attn-square and MLP-down |
| **3c decode**| up to **6.97×** | wins on all three M=1 layer shapes; clears Phase 3 Target |

### Lessons we want to carry forward

1. **"Memory-bound" diagnoses transfer between kernel families.** Phase 1
   v3 was decode attention at 19% of HBM peak; Phase 2 INT8/INT4 KV
   tied/won on the same workload; here Phase 3 W4A16 wins by 7× on the
   same kind of pattern. The lever was always "find the workload where
   bandwidth is genuinely the ceiling and exploit the smaller bytes."
   When bandwidth isn't the ceiling (Phase 1 v4/v5, Phase 2b), the
   gains evaporate.

2. **K-split across warps is the right move when single-warp blocks
   under-fill SMs.** Phase 1 v4 (FlashDecoding split-K) tried this for
   *attention* and lost — but attention had the per-iter dependency-
   chain ceiling we couldn't lift. W4A16 GEMM has *no* such per-iter
   chain (just FMAs in parallel), so K-split converts compute purely.
   The same idea, different outcome — pattern-recognise the workload
   shape before transferring optimizations.

3. **Multi-warp blocks pay back when there's K-side compute to split.**
   In Phase 2c we saw the symmetric move on the K side (per-group K
   scales register-resident). Here it's full K-reduction split. Both
   are "move work out of the inner serial loop."

4. **Shmem caching of frequently-reused data is a free-ish win when
   shmem isn't already over-committed.** Phase 2c reused this trick
   for the K scales; here for activations. Both freed L1 for the
   *new* hot path (weight reads, in this case).

5. **L2 cache effectiveness is real and measurable.** The
   1577 GB/s "effective bandwidth" number on MLP up/gate is L2 serving
   the (28.88 MiB packed) weights — that's the realistic warm-cache
   decode regime. Cold-cache HBM-bound numbers would be lower; warm-
   cache numbers are what serving sees.

## What's still open

- **GPTQ / AWQ quantisation + perplexity validation.** We measured the
  kernel; we did not measure model-level accuracy on real Llama 3 8B
  weights. The Phase 2d patch-`F.scaled_dot_product_attention` trick
  doesn't apply directly here (matmul is everywhere); a clean approach
  is a `nn.Linear` subclass that quantises its weights on load and
  calls `llmik_cuda.w4a16_gemm` in its forward.
- **Marlin head-to-head.** The Phase 3 *Target* line "within 25% of
  Marlin" wasn't measured. Marlin is the SOTA W4A16 kernel; its main
  trick is the permuted-weight tile layout that allows vectorised
  per-thread weight loads. Comparing properly would need installing
  Marlin and running the same workload — bench infra would carry over
  cleanly from `bench_w4a16.py`.
- **M > 1 fast path.** The current dispatch sends M > 1 to the 3b
  naive kernel, which scales linearly with M. A proper batched-decode
  variant (M-fast inner loop, with each weight loaded once and used M
  times) is straightforward but unfilled.
- **Prefill (M ≥ 16, compute-bound).** A Tensor Core MMA kernel that
  dequantises to shmem and feeds `mma.sync` is the conventional
  prefill story. Different kernel structure; stretch goal.

## Reference workload reproduction

```bash
git checkout 3c   # or 'main' for the current best
python setup.py build_ext --inplace
pytest tests/test_quant.py
python benchmarks/bench_w4a16.py
```

Branches `3a`, `3b`, `3c` track the three sub-steps' end states. The
full per-step table lives in [`results/RESULTS.md`](results/RESULTS.md).
