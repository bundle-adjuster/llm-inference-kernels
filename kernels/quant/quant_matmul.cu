#include "quant/quant_matmul.cuh"
#include "common/cuda_utils.cuh"

// W4A16 GEMM — Phase 3b naive kernel.
// Layout & contract: kernels/quant/quant_matmul.cuh
// Design: docs/03-quantized-matmul.md
//
// Targets the decode shape (M small, typically 1): the GEMM is
// memory-bound on weight traffic, and INT4 weights cut HBM bytes ~4×.
//
// Block geometry:
//   - One block per BLOCK_N output columns (BLOCK_N = 32).
//   - One thread per output column. Block = 1 warp = 32 threads.
//   - Each thread independently accumulates one (m, n) output element.
//
// Inner loop per (m, n) — the work each thread does:
//   for g in 0..n_groups-1:
//     scale = scales[g, n]                  (per-thread, fp16 -> float)
//     for p in 0..packs_per_group-1:
//       w_packed = weight_packed[g*packs_per_group + p, n]   (one uint32)
//       for i in 0..7:
//         nibble = (int32_t)(w_packed << (28 - i*4)) >> 28   (sign-extend)
//         k = (g*packs_per_group + p)*8 + i
//         acc += act[m, k] * (nibble * scale)
//   out[m, n] = acc (cast to fp16)
//
// Memory-traffic accounting (one kernel call):
//   - Each thread reads K act values per row m. Across BLOCK_N=32 threads
//     in a block, the act reads land on the same K addresses → L1 caches
//     them across the warp. Per-block act HBM traffic ≈ K · sizeof(half).
//   - Each thread reads K/8 packed weight uint32's. Across the warp, the
//     32 reads at iter k_pack are weight_packed[k_pack, n_base..n_base+31]
//     — 32 contiguous int32 values = one warp-wide 128-byte coalesced load.
//
// What's intentionally not done (saved for 3c):
//   - act caching in shared memory (the L1 hits are decent but suboptimal)
//   - vectorized weight loads (we could load 4 uint32 per thread per pop)
//   - scale broadcasting / smarter scale loads
//
// Roadmap (one commit + one RESULTS.md row per step):
//   3a  PyTorch reference + tests
//   3b  Naive CUDA kernel + correctness                          <-- here
//   3c  Optimization sweep for decode (act in shmem, vec loads, etc.)
//   3e  Benchmark vs torch.matmul + journey doc

namespace {
constexpr int BLOCK_N = 32;  // output columns per block = threads per block (one warp)
constexpr int PACK    = 8;   // 4-bit values packed per uint32
}  // namespace

__global__ void w4a16_gemm_naive_kernel(
    const half* __restrict__ act,
    const uint32_t* __restrict__ weight_packed,
    const half* __restrict__ scales,
    half* __restrict__ out,
    int M, int N, int K, int group_size) {

    const int n = blockIdx.x * BLOCK_N + threadIdx.x;
    if (n >= N) return;

    const int n_groups        = K / group_size;
    const int packs_per_group = group_size / PACK;

    // Outer loop over output rows m. For decode M=1 this runs once.
    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;

        for (int g = 0; g < n_groups; ++g) {
            // Per-(group, output-column) scale. Each thread owns a different
            // column, so each thread loads its own scale once per group.
            const float scale = __half2float(scales[g * N + n]);

            for (int p = 0; p < packs_per_group; ++p) {
                const int      k_pack   = g * packs_per_group + p;
                const uint32_t w_raw    = weight_packed[k_pack * N + n];
                const int32_t  signed_w = static_cast<int32_t>(w_raw);

                // Unpack 8 nibbles and FMA. The compiler should unroll
                // this fully (PACK=8 is a small constant); each iter is
                // a shift+arith-shift sign-extend, a load from act, and
                // an FMA into acc.
                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    // Shift the i-th nibble to bits [28..31] then arith
                    // shift right by 28 — PTX `shr.s32` sign-extends.
                    const int    nibble = (signed_w << (28 - i * 4)) >> 28;
                    const int    k      = k_pack * PACK + i;
                    const float  a      = __half2float(act[m * K + k]);
                    const float  w      = static_cast<float>(nibble) * scale;
                    acc += a * w;
                }
            }
        }

        out[m * N + n] = __float2half(acc);
    }
}

void launch_w4a16_gemm(
    const half* act, const uint32_t* weight_packed, const half* scales,
    half* out, int M, int N, int K, int group_size, cudaStream_t stream) {
    // Tail blocks handle the `n >= N` case via the in-kernel guard, so
    // N doesn't need to be a multiple of BLOCK_N. K must be a multiple
    // of group_size (and group_size of PACK=8) — checked at the binding.
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(BLOCK_N);
    w4a16_gemm_naive_kernel<<<grid, block, 0, stream>>>(
        act, weight_packed, scales, out, M, N, K, group_size);
    CUDA_CHECK_LAST();
}
