#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention — v1, online (streaming) softmax.
// Layout & contract: kernels/attention/fused_attention.cuh
//
// Algorithm: Milakov & Gimelshein 2018 (the running-max softmax at the heart of
// FlashAttention). One thread block per (batch, head); blockDim.x == head_dim.
// Each thread owns:
//   - one Q lane (head_dim/block lanes per thread = 1 here)
//   - one O lane (its private fp32 output accumulator)
//   - identical scalar running state (m, l) — same across threads in the block
//     because every thread sees the same broadcast s_j and updates identically.
//
// Per kv position j, in one pass:
//   partial = q_val * k[j, tid]
//   s_j     = scale · block_reduce_sum(partial)
//   m_new   = max(m, s_j)
//   alpha   = exp(m - m_new)              // rescaler for prior accumulators
//   p_j     = exp(s_j - m_new)
//   o_acc   = o_acc * alpha + p_j * v[j, tid]
//   l       = l     * alpha + p_j
//   m       = m_new
// At end:  out[tid] = o_acc / l
//
// fp16 in/out, fp32 accumulation. No score buffer in shmem → seqlen_kv is
// unbounded (limited only by HBM and timing).
//
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive, two-pass softmax
//   v1  online (streaming) softmax, single pass                       <-- here
//   v2  warp-level reductions (refinement of the dot-product reduce)
//   v3  vectorized 128-bit coalesced KV loads
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

}  // namespace

__global__ void decode_attention_kernel_v1(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale) {

    __shared__ float reduce_smem[32];   // one slot per warp for cross-warp dot-product reduce
    __shared__ float s_bcast;           // broadcast slot for the reduced s_j

    const int batch_idx   = blockIdx.x;
    const int head_idx    = blockIdx.y;
    const int tid         = threadIdx.x;
    const int bdim        = blockDim.x;     // = head_dim
    const int lane_id     = tid & 31;
    const int warp_id     = tid >> 5;
    const int n_warps     = bdim >> 5;

    // GQA: query head h reads kv head h / (n_heads / n_kv_heads).
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const half* q_ptr  = q + (batch_idx * n_heads    + head_idx)    * head_dim;
    const half* k_base = k + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half* v_base = v + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    half*       o_ptr  = out + (batch_idx * n_heads  + head_idx)    * head_dim;

    // Each thread owns one q lane in a register.
    const float q_val = __half2float(q_ptr[tid]);

    // Online-softmax running state.
    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state = 0.0f;

    for (int j = 0; j < seqlen_kv; ++j) {
        // ---- 1. block-wide dot: s_j = scale · Σ_d q[d] · k[j, d] ----
        float partial = q_val * __half2float(k_base[j * head_dim + tid]);
        partial = warp_reduce_sum(partial);
        if (lane_id == 0) reduce_smem[warp_id] = partial;
        __syncthreads();
        if (warp_id == 0) {
            float r = (lane_id < n_warps) ? reduce_smem[lane_id] : 0.0f;
            r = warp_reduce_sum(r);
            if (lane_id == 0) s_bcast = r * softmax_scale;
        }
        __syncthreads();
        const float s_j = s_bcast;

        // ---- 2. online softmax + output accumulator update ----
        // First iteration: m_state = -INF, so alpha = exp(-INF) = 0, o_state * 0 = 0,
        //   l_state * 0 = 0, and we initialize cleanly with the j=0 contribution.
        const float m_new = fmaxf(m_state, s_j);
        const float alpha = __expf(m_state - m_new);
        const float p_j   = __expf(s_j     - m_new);
        o_state = o_state * alpha + p_j * __half2float(v_base[j * head_dim + tid]);
        l_state = l_state * alpha + p_j;
        m_state = m_new;
    }

    o_ptr[tid] = __float2half(o_state / l_state);
}

void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    dim3 block(head_dim);
    dim3 grid(batch, n_heads);
    decode_attention_kernel_v1<<<grid, block, 0, stream>>>(
        q, k, v, out, batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale);
    CUDA_CHECK_LAST();
}
