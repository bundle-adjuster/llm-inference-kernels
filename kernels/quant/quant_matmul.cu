#include "quant/quant_matmul.cuh"
#include "common/cuda_utils.cuh"
#include <mma.h>

// W4A16 GEMM kernels.
// Layout & contract: kernels/quant/quant_matmul.cuh
// Design: docs/03-quantized-matmul.md
//
// Four kernels in this file:
//   v0 naive  (Phase 3b) — one warp per BLOCK_N=32 output tile; outer
//             M loop. Catch-all; used when M > 16 in the launcher.
//   v1 decode (Phase 3c) — four-warp block per BLOCK_N=32 tile, K split
//             across the 4 warps with a tiny shmem combine at the end;
//             act cached in shared memory. Fast path for M = 1.
//   v2 batched-decode (Phase 5) — same K-split-across-warps as v1, but
//             each thread accumulates a length-BLOCK_M=16 vector of fp32
//             partials. Per-warp act tile in shmem. Used for 2 <= M <= 16,
//             scalar FP32 FMA inner loop.
//   v3 tensor-core (Phase 6) — same M=16 BLOCK_M but inner loop uses
//             `mma.sync` via wmma fragments (16x16x16 fp16->fp32). Each
//             warp owns BLOCK_N=16 output columns; one warp = one MMA
//             tile. Closes the cuBLAS gap at M=16 by using tensor cores
//             on the dequantized weight stream.
//
// Launcher dispatch:
//   M == 1            -> v1 decode
//   M in [2, 16]      -> v3 tensor-core (v2 is kept for reference / comparison)
//   M  > 16           -> v0 naive

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
// v3: tensor-core W4A16 GEMM. Phase 6.
//
// Same M=16 batched-decode shape as v2, but the inner accumulate loop uses
// `mma.sync` via the wmma C++ API. Each warp owns BLOCK_N=16 output cols
// (BLOCK_N_PER_WARP); 4 warps per block -> BLOCK_N=64 output cols per block.
// No K-split across warps: each warp processes the full K independently
// across its own col tile.
//
// Per K group (group_size=128 K positions):
//   1. Cooperative load act tile [BLOCK_M=16, group_size] into shmem (shared
//      across all warps in the block).
//   2. Each warp dequantizes its [group_size, BLOCK_N_PER_WARP] weight tile
//      (int4 + per-channel-groupwise scale -> fp16) into its slice of
//      shared memory.
//   3. Each warp loops over the group_size axis in WMMA_K=16 chunks (8 iters
//      per group). One `mma.sync.m16n16k16.row.row.f32.f16.f16.f32` per
//      inner step, accumulating into the warp's c_frag (fp32, [16, 16]).
//   4. After all K groups, convert c_frag to fp16 and store to global.
//
// Why this beats v2 at M=16: tensor cores deliver fp16 MMA throughput
// (~165 TFLOPs on sm_89) vs scalar fp32 FMA (~82 TFLOPs). At M=16 the
// per-block work is large enough that the FMA throughput, not memory,
// is the bottleneck for the int4 weight stream (v2 was compute-bound at
// 117 GB/s on int4 weights vs cuBLAS at 812 GB/s on fp16). Tensor cores
// unblock that compute ceiling.
//
// Shared memory layout (dynamic):
//   [BLOCK_M  * group_size           half ]  act_smem (shared across warps)
//   [WARPS    * group_size * BLOCK_N_PER_WARP  half ]  weight_smem (per-warp tiles)
//   [BLOCK_M  * BLOCK_N              half ]  output_smem (reuses act after K loop)
// ============================================================================

using namespace nvcuda::wmma;

constexpr int TC_BLOCK_M       = 16;
constexpr int TC_BLOCK_N_PER_WARP = 16;
constexpr int TC_WMMA_M        = 16;
constexpr int TC_WMMA_N        = 16;
constexpr int TC_WMMA_K        = 16;
constexpr int TC_WARPS         = 4;
constexpr int TC_BLOCK_N       = TC_WARPS * TC_BLOCK_N_PER_WARP;  // = 64

__global__ void w4a16_gemm_tc_kernel(
    const half* __restrict__ act,                  // [M, K]
    const uint32_t* __restrict__ weight_packed,    // [K/PACK, N]
    const half* __restrict__ scales,               // [n_groups, N]
    half* __restrict__ out,                        // [M, N]
    int M, int N, int K, int group_size) {

    constexpr int THREADS = TC_WARPS * 32;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid & 31;

    extern __shared__ char smem[];
    half* act_smem    = reinterpret_cast<half*>(smem);                       // [BLOCK_M, group_size]
    half* weight_smem = act_smem + TC_BLOCK_M * group_size;                  // [WARPS][group_size, BLOCK_N_PER_WARP]

    const int block_n_start = blockIdx.x * TC_BLOCK_N;
    const int warp_n_start  = block_n_start + warp_id * TC_BLOCK_N_PER_WARP;

    // Per-warp accumulator. fp32 to avoid overflow across many K positions.
    fragment<accumulator, TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    const int n_groups = K / group_size;

    half* my_weight_smem = weight_smem
        + warp_id * group_size * TC_BLOCK_N_PER_WARP;

    for (int g = 0; g < n_groups; ++g) {
        const int k_start = g * group_size;

        // (1) Cooperative load act tile [BLOCK_M, group_size]
        for (int idx = tid; idx < TC_BLOCK_M * group_size; idx += THREADS) {
            const int m     = idx / group_size;
            const int k_off = idx % group_size;
            act_smem[m * group_size + k_off] = (m < M)
                ? act[m * K + k_start + k_off]
                : __float2half(0.0f);
        }

        // (2) Each warp dequantizes its [group_size, BLOCK_N_PER_WARP] weight tile
        const int tile_elems = group_size * TC_BLOCK_N_PER_WARP;
        for (int idx = lane; idx < tile_elems; idx += 32) {
            const int k             = idx / TC_BLOCK_N_PER_WARP;
            const int n_col_in_warp = idx % TC_BLOCK_N_PER_WARP;
            const int n_col         = warp_n_start + n_col_in_warp;
            const bool valid        = (n_col < N);

            const int      k_pack    = (k_start + k) / PACK;
            const int      k_in_pack = k % PACK;
            const uint32_t packed    = valid
                ? weight_packed[k_pack * N + n_col]
                : 0u;
            const int      nibble    = (static_cast<int32_t>(packed)
                                        << (28 - k_in_pack * 4)) >> 28;
            const float    scale_v   = valid
                ? __half2float(scales[g * N + n_col])
                : 0.0f;
            my_weight_smem[k * TC_BLOCK_N_PER_WARP + n_col_in_warp] =
                __float2half(static_cast<float>(nibble) * scale_v);
        }
        __syncthreads();

        // (3) Inner loop: BLOCK_K=group_size processed in WMMA_K=16 chunks
        fragment<matrix_a, TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, half, row_major> b_frag;

        const int wmma_steps = group_size / TC_WMMA_K;
        #pragma unroll 1
        for (int wk = 0; wk < wmma_steps; ++wk) {
            // act tile is laid out [BLOCK_M, group_size] row-major; ldm = group_size
            load_matrix_sync(a_frag, act_smem + wk * TC_WMMA_K, group_size);
            // weight tile is laid out [group_size, BLOCK_N_PER_WARP] row-major; ldm = BLOCK_N_PER_WARP
            load_matrix_sync(b_frag,
                             my_weight_smem + wk * TC_WMMA_K * TC_BLOCK_N_PER_WARP,
                             TC_BLOCK_N_PER_WARP);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();  // before next group writes weight_smem
    }

    // (4) Convert fp32 accumulator to fp16 and stage in shmem (reuses act_smem
    // since the K loop is done)
    half* output_smem = act_smem;  // [WARPS * BLOCK_M * BLOCK_N_PER_WARP] half — but layout is
                                   // per-warp [BLOCK_M, BLOCK_N_PER_WARP], stacked along warp_id.
    fragment<accumulator, TC_WMMA_M, TC_WMMA_N, TC_WMMA_K, half> c_out_frag;
    #pragma unroll
    for (int i = 0; i < c_frag.num_elements; ++i) {
        c_out_frag.x[i] = __float2half(c_frag.x[i]);
    }
    store_matrix_sync(
        output_smem + warp_id * TC_BLOCK_M * TC_BLOCK_N_PER_WARP,
        c_out_frag, TC_BLOCK_N_PER_WARP, mem_row_major);
    __syncthreads();

    // Cooperative write to out[M, N] — only rows in [0, M).
    for (int idx = tid; idx < M * TC_BLOCK_N; idx += THREADS) {
        const int m            = idx / TC_BLOCK_N;
        const int n_in_block   = idx % TC_BLOCK_N;
        const int warp         = n_in_block / TC_BLOCK_N_PER_WARP;
        const int n_in_warp    = n_in_block % TC_BLOCK_N_PER_WARP;
        const int n_col        = block_n_start + n_in_block;
        if (n_col < N) {
            out[m * N + n_col] =
                output_smem[warp * TC_BLOCK_M * TC_BLOCK_N_PER_WARP
                            + m * TC_BLOCK_N_PER_WARP + n_in_warp];
        }
    }
}


// ============================================================================
// Launcher: dispatch
//   M == 1                   → v1 decode (3c)
//   1 < M <= 16              → v3 tensor-core (Phase 6)  -- v2 retired from dispatch
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
    } else if (M <= TC_BLOCK_M) {
        // Batched decode with tensor cores (Phase 6).
        constexpr int THREADS = TC_WARPS * 32;
        dim3 grid((N + TC_BLOCK_N - 1) / TC_BLOCK_N);
        dim3 block(THREADS);
        // Shmem: act_smem (also reused as output_smem) + weight_smem
        //  act:    BLOCK_M * group_size
        //  output: WARPS * BLOCK_M * BLOCK_N_PER_WARP = BLOCK_M * BLOCK_N
        //          (same total size — both contiguous from the same pointer)
        //  weight: WARPS * group_size * BLOCK_N_PER_WARP
        const size_t act_or_output_bytes =
            static_cast<size_t>(TC_BLOCK_M) * TC_BLOCK_N * sizeof(half);
        const size_t initial_act_bytes =
            static_cast<size_t>(TC_BLOCK_M) * group_size * sizeof(half);
        const size_t smem_bytes =
            (act_or_output_bytes > initial_act_bytes ? act_or_output_bytes : initial_act_bytes)
            + static_cast<size_t>(TC_WARPS) * group_size * TC_BLOCK_N_PER_WARP * sizeof(half);
        w4a16_gemm_tc_kernel<<<grid, block, smem_bytes, stream>>>(
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
