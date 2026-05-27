#include "quant/quant_matmul.cuh"
#include "common/cuda_utils.cuh"

// W4A16 GEMM kernels.
// Layout & contract: kernels/quant/quant_matmul.cuh
// Design: docs/03-quantized-matmul.md
//
// Two kernels in this file:
//   v0 naive  (Phase 3b) — one warp per BLOCK_N=32 output tile; outer
//             M loop. Catch-all; used when M > 1 in the launcher.
//   v1 decode (Phase 3c) — four-warp block per BLOCK_N=32 tile, K split
//             across the 4 warps with a tiny shmem combine at the end;
//             act cached in shared memory. Fast path for M = 1.
//
// The launcher dispatches: M == 1 → v1 decode, else → v0 naive.

namespace {
constexpr int BLOCK_N = 32;       // output columns per block (one warp's worth of lanes)
constexpr int PACK    = 8;        // 4-bit values per packed uint32
}  // namespace


// ============================================================================
// v0: naive W4A16 GEMM. Phase 3b. One warp per output tile. Outer-M loop.
// ============================================================================

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

    for (int m = 0; m < M; ++m) {
        float acc = 0.0f;

        for (int g = 0; g < n_groups; ++g) {
            const float scale = __half2float(scales[g * N + n]);

            for (int p = 0; p < packs_per_group; ++p) {
                const int      k_pack   = g * packs_per_group + p;
                const uint32_t w_raw    = weight_packed[k_pack * N + n];
                const int32_t  signed_w = static_cast<int32_t>(w_raw);

                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    const int   nibble = (signed_w << (28 - i * 4)) >> 28;
                    const int   k      = k_pack * PACK + i;
                    const float a      = __half2float(act[m * K + k]);
                    const float w      = static_cast<float>(nibble) * scale;
                    acc += a * w;
                }
            }
        }

        out[m * N + n] = __float2half(acc);
    }
}


// ============================================================================
// v1: decode-optimized W4A16 GEMM. Phase 3c. M = 1 only.
//
// Block geometry:
//   - 128 threads (4 warps).
//   - BLOCK_N=32 output columns per block. Each lane (within any warp)
//     owns one output column n = block_n_base + lane.
//   - K split across the 4 warps. Each warp processes K / 4 of the
//     reduction for all 32 columns, accumulating into one fp32 per lane.
//   - After the K loop: a small shmem-based reduction sums the 4
//     per-warp partials per column. Lane writes the final fp16 output.
//
// Activations live in shared memory for the whole K loop. With M=1 and
// K = 4096..14336, that's 8..28 KiB of shmem — well under the 100 KiB/SM
// limit. Coalesced loads across the 128 threads pull act into shmem at
// kernel start with one __syncthreads().
//
// Per warp: groups_per_warp = (K / group_size) / 4. For K=4096,
// group_size=128: 32/4 = 8 groups per warp.
//
// Coalescing: at iter k_pack, the 32 threads of each warp load
// weight_packed[k_pack, n_base..n_base+31] — one warp-wide 128-byte
// coalesced load. Threads in different warps (same lane) load the same
// columns but for different k_pack — addresses differ by N, so different
// L1 lines.
//
// Why this beats v0 for M=1:
//   - At small N (=4096): v0 had only 128 1-warp blocks ≈ 1 warp/SM
//     across the 128 SMs. v1 has 128 4-warp blocks ≈ 4 warps/SM,
//     meaningful for latency hiding.
//   - At large K (=14336): v0's per-thread sequential K work is K=14336;
//     v1 splits to K/4 = 3584 per thread.
//   - Act cache: v0 relied on L1 for act reuse across threads in a
//     warp. v1's shmem cache is faster and frees L1 for weight traffic.

__global__ void w4a16_gemm_decode_kernel(
    const half* __restrict__ act,
    const uint32_t* __restrict__ weight_packed,
    const half* __restrict__ scales,
    half* __restrict__ out,
    int N, int K, int group_size) {

    constexpr int WARPS   = 4;
    constexpr int THREADS = WARPS * 32;   // = 128

    // Dynamic shmem layout: [K fp16 act_smem ][ WARPS*BLOCK_N float partials ].
    extern __shared__ char smem[];
    half*  act_smem     = reinterpret_cast<half*>(smem);
    float* partials_smem = reinterpret_cast<float*>(act_smem + K);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid & 31;

    const int n = blockIdx.x * BLOCK_N + lane;
    const bool valid = (n < N);

    // Cooperative load of act[0, 0..K) into shmem.
    // 128 threads × ceil(K / 128) iters each.
    for (int k = tid; k < K; k += THREADS) {
        act_smem[k] = act[k];
    }
    __syncthreads();

    // K-split across warps: each warp handles K / 4 of the reduction.
    const int n_groups        = K / group_size;
    const int packs_per_group = group_size / PACK;
    const int groups_per_warp = n_groups / WARPS;
    const int g_start         = warp_id * groups_per_warp;
    const int g_end           = g_start + groups_per_warp;

    float acc = 0.0f;
    if (valid) {
        for (int g = g_start; g < g_end; ++g) {
            const float scale = __half2float(scales[g * N + n]);

            for (int p = 0; p < packs_per_group; ++p) {
                const int      k_pack   = g * packs_per_group + p;
                const uint32_t w_raw    = weight_packed[k_pack * N + n];
                const int32_t  signed_w = static_cast<int32_t>(w_raw);

                #pragma unroll
                for (int i = 0; i < PACK; ++i) {
                    const int   nibble = (signed_w << (28 - i * 4)) >> 28;
                    const int   k      = k_pack * PACK + i;
                    const float a      = __half2float(act_smem[k]);
                    const float w      = static_cast<float>(nibble) * scale;
                    acc += a * w;
                }
            }
        }
    }

    // Stage per-warp partial. Even invalid lanes write 0 to keep the
    // shmem layout dense and the __syncthreads contract simple.
    partials_smem[warp_id * BLOCK_N + lane] = acc;
    __syncthreads();

    // Warp 0 sums the 4 partials for its column and writes the output.
    if (warp_id == 0 && valid) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            total += partials_smem[w * BLOCK_N + lane];
        }
        out[n] = __float2half(total);
    }
}


// ============================================================================
// v2: batched-decode W4A16 GEMM. Phase 5. The M=1 decode kernel re-reads
// the entire weight matrix for each of M=batch output rows when M > 1
// (because the launcher routes to v0 naive, which has no batch parallelism).
// At batch=16 (the locked Phase 4 e2e workload) this read amplification
// gave a 4c regression — see docs/04-end-to-end-integration-journey.md.
//
// Fix: same K-split-across-warps pattern as v1 decode, but each thread now
// accumulates a vector of M fp32 partials instead of a scalar. The packed
// weight + scale are loaded **once** per K position and reused across M.
// Per-warp activation tile [BLOCK_M, group_size] cached in shmem so each
// inner loop reads act from L1-fast shmem, not global.
//
// Design choices:
//   - BLOCK_M = 16. Matches Llama 3 e2e workload (batch=16). Larger BLOCK_M
//     would need K-tiling and a bigger inner loop; smaller would leave
//     batch parallelism on the table. M < 16 → unused accumulators (zeros);
//     M > 16 → launcher falls back to naive.
//   - BLOCK_N = 32. Same as v1; one warp's worth of output columns.
//   - 4 warps per block. Same K-split-across-warps as v1, so this kernel
//     is essentially v1 with the inner scalar accumulator replaced by
//     a length-M vector.
//   - act_smem per warp: [BLOCK_M, group_size] fp16. Loaded once per K-group
//     iteration; 4 KB per warp × 4 warps = 16 KB total. Fits easily.
//
// Shared memory: WARPS * BLOCK_M * group_size * sizeof(half)
//              + WARPS * BLOCK_M * BLOCK_N * sizeof(float)   (partials)
// At group_size=128, BLOCK_M=16, BLOCK_N=32, WARPS=4: 16 KB + 8 KB = 24 KB.
// ============================================================================

constexpr int BATCHED_DECODE_BLOCK_M = 16;

__global__ void w4a16_gemm_batched_decode_kernel(
    const half* __restrict__ act,            // [M, K]
    const uint32_t* __restrict__ weight_packed,  // [K/PACK, N]
    const half* __restrict__ scales,         // [n_groups, N]
    half* __restrict__ out,                  // [M, N]
    int M, int N, int K, int group_size) {

    constexpr int WARPS       = 4;
    constexpr int THREADS     = WARPS * 32;
    constexpr int BLOCK_M     = BATCHED_DECODE_BLOCK_M;

    extern __shared__ char smem[];
    // [WARPS][BLOCK_M][group_size] act_smem — each warp's slice of K rows
    half* act_smem = reinterpret_cast<half*>(smem);
    // [WARPS][BLOCK_M][BLOCK_N] float partials — cross-warp reduce at the end
    float* partials_smem = reinterpret_cast<float*>(
        act_smem + WARPS * BLOCK_M * group_size);

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid & 31;

    const int n = blockIdx.x * BLOCK_N + lane;
    const bool valid = (n < N);

    // Per-thread accumulators — one per M row of this thread's column.
    float acc[BLOCK_M];
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) acc[m] = 0.0f;

    // K split: each warp handles n_groups / WARPS groups.
    const int n_groups        = K / group_size;
    const int packs_per_group = group_size / PACK;
    const int groups_per_warp = n_groups / WARPS;
    const int g_start         = warp_id * groups_per_warp;
    const int g_end           = g_start + groups_per_warp;

    half* my_act_smem = act_smem + warp_id * BLOCK_M * group_size;

    for (int g = g_start; g < g_end; ++g) {
        // Cooperative load of this group's act tile [M, group_size] into shmem.
        // 32 lanes load M*group_size = 16*128 = 2048 halves → 64 per lane.
        const int k_start = g * group_size;
        const int tile_size = BLOCK_M * group_size;
        for (int idx = lane; idx < tile_size; idx += 32) {
            const int m = idx / group_size;
            const int k_off = idx % group_size;
            // Bounds: pad unused M rows with 0 (acc unaffected since acc[m] for
            // m >= M is initialized to 0 and never read for output).
            const half a = (m < M)
                ? act[m * K + k_start + k_off]
                : __float2half(0.0f);
            my_act_smem[m * group_size + k_off] = a;
        }
        __syncwarp();

        const float scale = valid
            ? __half2float(scales[g * N + n])
            : 0.0f;

        for (int p = 0; p < packs_per_group; ++p) {
            const int      k_pack   = g * packs_per_group + p;
            const uint32_t w_raw    = valid
                ? weight_packed[k_pack * N + n]
                : 0u;
            const int32_t  signed_w = static_cast<int32_t>(w_raw);

            #pragma unroll
            for (int i = 0; i < PACK; ++i) {
                const int   nibble  = (signed_w << (28 - i * 4)) >> 28;
                const float w_val   = static_cast<float>(nibble) * scale;
                const int   k_in_g  = p * PACK + i;

                #pragma unroll
                for (int m = 0; m < BLOCK_M; ++m) {
                    const float a = __half2float(
                        my_act_smem[m * group_size + k_in_g]);
                    acc[m] += a * w_val;
                }
            }
        }
    }

    // Stage per-warp partials.
    #pragma unroll
    for (int m = 0; m < BLOCK_M; ++m) {
        partials_smem[(warp_id * BLOCK_M + m) * BLOCK_N + lane] = acc[m];
    }
    __syncthreads();

    // Warp 0 reduces across WARPS for each M row of its column.
    if (warp_id == 0 && valid) {
        for (int m = 0; m < M; ++m) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) {
                total += partials_smem[(w * BLOCK_M + m) * BLOCK_N + lane];
            }
            out[m * N + n] = __float2half(total);
        }
    }
}


// ============================================================================
// Launcher: dispatch
//   M == 1                   → v1 decode (3c)
//   1 < M <= BLOCK_M (=16)   → v2 batched-decode (Phase 5)
//   M > 16                   → v0 naive (3b)   — catch-all for prefill etc.
// ============================================================================

void launch_w4a16_gemm(
    const half* act, const uint32_t* weight_packed, const half* scales,
    half* out, int M, int N, int K, int group_size, cudaStream_t stream) {

    if (M == 1) {
        // Fast path for single-stream decode.
        constexpr int WARPS   = 4;
        constexpr int THREADS = WARPS * 32;
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
        dim3 block(THREADS);
        const size_t smem_bytes =
            static_cast<size_t>(K) * sizeof(half)
            + WARPS * BLOCK_N * sizeof(float);
        w4a16_gemm_decode_kernel<<<grid, block, smem_bytes, stream>>>(
            act, weight_packed, scales, out, N, K, group_size);
    } else if (M <= BATCHED_DECODE_BLOCK_M) {
        // Batched decode (Phase 5).
        constexpr int WARPS   = 4;
        constexpr int THREADS = WARPS * 32;
        constexpr int BLOCK_M = BATCHED_DECODE_BLOCK_M;
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
        dim3 block(THREADS);
        const size_t smem_bytes =
            static_cast<size_t>(WARPS) * BLOCK_M * group_size * sizeof(half)
            + static_cast<size_t>(WARPS) * BLOCK_M * BLOCK_N * sizeof(float);
        w4a16_gemm_batched_decode_kernel<<<grid, block, smem_bytes, stream>>>(
            act, weight_packed, scales, out, M, N, K, group_size);
    } else {
        // Catch-all for M > BLOCK_M (e.g. prefill at M=8192).
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N);
        dim3 block(BLOCK_N);
        w4a16_gemm_naive_kernel<<<grid, block, 0, stream>>>(
            act, weight_packed, scales, out, M, N, K, group_size);
    }
    CUDA_CHECK_LAST();
}
