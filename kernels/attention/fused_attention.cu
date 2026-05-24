#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention — v3, single-warp block + vectorized KV loads.
// Layout & contract: kernels/attention/fused_attention.cuh
//
// Algorithm: same Milakov–Gimelshein online softmax as v1/v2. Two changes
// from v2:
//
//   1. Single-warp blocks (blockDim.x == 32). Each thread now owns
//      VEC = head_dim / 32 = 4 d-lanes (vs 1 d-lane with the v2 128-thread
//      block). The whole dot-product reduction now fits inside one warp, so
//      every `__syncthreads()` and every cross-warp shmem slot from v2 is
//      gone. The block reduce is just `warp_reduce_sum` on a 4-element
//      per-thread partial.
//
//   2. Vectorized 64-bit KV loads. Each thread loads VEC halves at a time
//      via uint2 + __half22float2 (LDG.E.64). One warp-wide K load is now
//      32 × 8 = 256 bytes (= head_dim halves = one full j) in a single
//      instruction, where v2 needed four 64-byte warp loads.
//
// Trade-off: per-block thread count drops 4×, so per-SM occupancy drops
// from 12 blocks (48 warps, full) to ~16 blocks (16 warps, ~33%). The
// per-warp load throughput from vectorization has to compensate for the
// lost latency-hiding warps.
//
// Per kv position j, in one pass:
//   k_v, v_v = load_half4_as_float4 of K[j, tid*VEC ..] and V[j, tid*VEC ..]
//   partial  = Σ_d q_v[d] · k_v[d]                            (4 FMAs per thread)
//   s_j      = scale · warp_reduce_sum(partial)               (one shfl tree;
//                                                              every lane gets s_j)
//   m_new    = max(m, s_j)
//   alpha    = exp(m - m_new)
//   p_j      = exp(s_j - m_new)
//   o_v[d]   = o_v[d] · alpha + p_j · v_v[d]                  (4 FMAs per thread)
//   l        = l · alpha + p_j
//   m        = m_new
// At end:  out[tid*VEC ..] = o_v / l                          (vectorized store)
//
// fp16 in/out, fp32 accumulation.
//
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive, two-pass softmax
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions (single-sync block reduce)
//   v3  vectorized KV loads (single-warp block, 64-bit per thread)    <-- here
//   v4  split-K over the KV sequence + combine (FlashDecoding)
//   v5  cp.async double-buffering of KV tiles

namespace {

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

// 8-byte aligned 4-half load → float4. Compiles to one LDG.E.64.
__device__ __forceinline__ float4 load_half4_as_float4(const half* ptr) {
    const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    half2 lo, hi;
    *reinterpret_cast<unsigned int*>(&lo) = raw.x;
    *reinterpret_cast<unsigned int*>(&hi) = raw.y;
    const float2 f_lo = __half22float2(lo);
    const float2 f_hi = __half22float2(hi);
    return make_float4(f_lo.x, f_lo.y, f_hi.x, f_hi.y);
}

// Inverse of the above. Compiles to one STG.E.64.
__device__ __forceinline__ void store_float4_as_half4(half* ptr, const float4 v) {
    const half2 lo = __floats2half2_rn(v.x, v.y);
    const half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const unsigned int*>(&lo);
    raw.y = *reinterpret_cast<const unsigned int*>(&hi);
    *reinterpret_cast<uint2*>(ptr) = raw;
}

}  // namespace

__global__ void decode_attention_kernel_v3(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale) {

    constexpr int VEC = 4;  // halves per thread per load (8 bytes)

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;   // 0..31; also lane_id

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const half* q_ptr  = q + (batch_idx * n_heads    + head_idx)    * head_dim + tid * VEC;
    const half* k_base = k + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half* v_base = v + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    half*       o_ptr  = out + (batch_idx * n_heads  + head_idx)    * head_dim + tid * VEC;

    // Load q (4 halves per thread).
    const float4 q_v = load_half4_as_float4(q_ptr);

    float4 o_v   = {0.0f, 0.0f, 0.0f, 0.0f};
    float  m_state = -INFINITY;
    float  l_state = 0.0f;

    for (int j = 0; j < seqlen_kv; ++j) {
        // Vectorized K + V loads at the top of the iteration. V is unused
        // until after the reduce and softmax update; its latency hides behind
        // both.
        const float4 k_v = load_half4_as_float4(k_base + j * head_dim + tid * VEC);
        const float4 v_v = load_half4_as_float4(v_base + j * head_dim + tid * VEC);

        // Local dot product across this thread's 4 d-lanes.
        float partial = q_v.x * k_v.x + q_v.y * k_v.y
                      + q_v.z * k_v.z + q_v.w * k_v.w;
        // Single-warp block reduce — every lane gets the same s_j.
        partial = warp_reduce_sum(partial);
        const float s_j = partial * softmax_scale;

        // Online softmax + output accumulator update.
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
    dim3 block(32);             // 1 warp; head_dim handled via 4-way per-thread vec.
    dim3 grid(batch, n_heads);
    decode_attention_kernel_v3<<<grid, block, 0, stream>>>(
        q, k, v, out, batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale);
    CUDA_CHECK_LAST();
}
