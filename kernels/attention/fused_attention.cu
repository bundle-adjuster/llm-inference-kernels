#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"
#include <cuda_pipeline.h>

// Decode attention — v5, cp.async double-buffered KV tiles.
// Layout & contract: kernels/attention/fused_attention.cuh
//
// v3's V prefetch papers over load latency via compiler scheduling: nvcc
// hoists one load before consumption, hiding it behind the per-`j`
// warp_reduce_sum. v5 makes that pipelining *explicit* with cp.async:
// each tile (K[j, :] + V[j, :]) is staged into a 2-slot dynamic shmem
// buffer via cp.async, fired and forgotten. While iter j consumes its
// tile, iter j+1's tile is already in flight to the other shmem slot.
//
//   prime:    issue load(tile_0 -> slot 0)
//   for j in [0, seqlen_kv):
//     if j+1 < seqlen_kv: issue load(tile_{j+1} -> slot ((j+1)&1))
//     __pipeline_wait_prior(N)        // N=1 with prefetch, 0 on last iter
//     read slot (j&1) from shmem; warp_reduce_sum; online softmax + V FMA
//
// cp.async bypasses L1 (goes L2 -> shmem). For decode attention, K+V per
// head is ~2 MB while L1 is 128 KB, so L1 wasn't catching anything anyway.
// Shmem cost: 2 stages × head_dim × 2 (K+V) × 2 B = ~1 KB total.
//
// Per-thread invariant: each thread cp.asyncs only its own 8-byte slot
// and reads back only its own slot. No cross-thread shmem traffic in the
// pipeline, so no __syncwarp() needed between wait and consume.
//
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive, two-pass softmax
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions (single-sync block reduce)
//   v3  vectorized KV loads (single-warp block, 64-bit per thread)
//   v4  split-K over the KV sequence + combine (FlashDecoding) — reverted
//   v5  cp.async double-buffering of KV tiles                       <-- here

namespace {

constexpr int VEC        = 4;   // halves per thread per vec load (8 B)
constexpr int NUM_STAGES = 2;   // double buffer

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float4 load_half4_as_float4(const half* ptr) {
    const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    half2 lo, hi;
    *reinterpret_cast<unsigned int*>(&lo) = raw.x;
    *reinterpret_cast<unsigned int*>(&hi) = raw.y;
    const float2 f_lo = __half22float2(lo);
    const float2 f_hi = __half22float2(hi);
    return make_float4(f_lo.x, f_lo.y, f_hi.x, f_hi.y);
}

__device__ __forceinline__ void store_float4_as_half4(half* ptr, const float4 v) {
    const half2 lo = __floats2half2_rn(v.x, v.y);
    const half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const unsigned int*>(&lo);
    raw.y = *reinterpret_cast<const unsigned int*>(&hi);
    *reinterpret_cast<uint2*>(ptr) = raw;
}

}  // namespace

__global__ void decode_attention_kernel_v5(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale) {

    extern __shared__ half kv_smem[];
    half* k_buf = kv_smem;                                // [NUM_STAGES][head_dim]
    half* v_buf = kv_smem + NUM_STAGES * head_dim;        // [NUM_STAGES][head_dim]

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;   // 0..31; also lane_id

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const half* q_ptr  = q + (batch_idx * n_heads    + head_idx)    * head_dim + tid * VEC;
    const half* k_base = k + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half* v_base = v + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    half*       o_ptr  = out + (batch_idx * n_heads  + head_idx)    * head_dim + tid * VEC;

    const float4 q_v = load_half4_as_float4(q_ptr);

    float4 o_v     = {0.0f, 0.0f, 0.0f, 0.0f};
    float  m_state = -INFINITY;
    float  l_state = 0.0f;

    // Prime the pipeline: issue tile 0 -> slot 0.
    __pipeline_memcpy_async(k_buf + 0 * head_dim + tid * VEC,
                            k_base + 0 * head_dim + tid * VEC, 8);
    __pipeline_memcpy_async(v_buf + 0 * head_dim + tid * VEC,
                            v_base + 0 * head_dim + tid * VEC, 8);
    __pipeline_commit();

    for (int j = 0; j < seqlen_kv; ++j) {
        const int cur  = j & 1;
        const int next = (j + 1) & 1;

        if (j + 1 < seqlen_kv) {
            // Prefetch tile j+1 into slot 'next'.
            __pipeline_memcpy_async(k_buf + next * head_dim + tid * VEC,
                                    k_base + (j + 1) * head_dim + tid * VEC, 8);
            __pipeline_memcpy_async(v_buf + next * head_dim + tid * VEC,
                                    v_base + (j + 1) * head_dim + tid * VEC, 8);
            __pipeline_commit();
            // Wait until 1 group remains: the j+1 prefetch can stay in flight,
            // but tile j's load (committed last iter / prime) must be done.
            __pipeline_wait_prior(1);
        } else {
            // Last iter: no prefetch. Drain to 0 outstanding.
            __pipeline_wait_prior(0);
        }
        // Each thread reads only its own slot from shmem — no cross-thread
        // shmem traffic, so no __syncwarp needed.

        const float4 k_v = load_half4_as_float4(k_buf + cur * head_dim + tid * VEC);
        const float4 v_v = load_half4_as_float4(v_buf + cur * head_dim + tid * VEC);

        // Same body as v3: dot product, warp reduce, online softmax + FMA.
        float partial = q_v.x * k_v.x + q_v.y * k_v.y
                      + q_v.z * k_v.z + q_v.w * k_v.w;
        partial = warp_reduce_sum(partial);
        const float s_j = partial * softmax_scale;

        const float m_new = fmaxf(m_state, s_j);
        const float alpha = __expf(m_state - m_new);
        const float p_j   = __expf(s_j     - m_new);
        o_v.x = o_v.x * alpha + p_j * v_v.x;
        o_v.y = o_v.y * alpha + p_j * v_v.y;
        o_v.z = o_v.z * alpha + p_j * v_v.z;
        o_v.w = o_v.w * alpha + p_j * v_v.w;
        l_state = l_state * alpha + p_j;
        m_state = m_new;
    }

    const float inv_l = 1.0f / l_state;
    o_v.x *= inv_l;  o_v.y *= inv_l;  o_v.z *= inv_l;  o_v.w *= inv_l;
    store_float4_as_half4(o_ptr, o_v);
}

void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch, n_heads);
    // K + V, each double-buffered, fp16. ~1 KB at head_dim=128.
    const size_t smem_bytes = NUM_STAGES * head_dim * sizeof(half) * 2;
    decode_attention_kernel_v5<<<grid, block, smem_bytes, stream>>>(
        q, k, v, out, batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale);
    CUDA_CHECK_LAST();
}
