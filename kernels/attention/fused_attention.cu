#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention — v2, online softmax with single-sync block reduce.
// Layout & contract: kernels/attention/fused_attention.cuh
//
// Algorithm: same Milakov–Gimelshein online softmax as v1. The change is
// in the per-`j` cross-warp synchronization. v1 had two `__syncthreads()`
// per iteration: one after each warp wrote its partial to shmem, and one
// after warp 0 broadcast `s_j`. v2 has *one*.
//
// Two changes make that work:
//   1. All warps redundantly do the final cross-warp reduce. After the
//      sync, every thread reads `reduce_smem[buf][lane_id]` (masking
//      lane_id >= n_warps to 0), runs a warp shfl, and obtains the same
//      block-wide partial. No more `s_bcast` write/read; no broadcast sync.
//   2. `reduce_smem` is double-buffered on `j & 1`. With a single buffer,
//      iter j+1's write would race iter j's still-in-flight reads in slower
//      warps. With two buffers, iter j+1 writes to slot ((j+1) & 1) while
//      iter j reads slot (j & 1) — different memory. The hazard between
//      iter j's read and iter j+2's write (same slot) is gated by iter j+1's
//      sync, so it's safe.
//
// Per kv position j, in one pass:
//   k_j, v_j  = K[j, tid], V[j, tid]                          (loads issued together;
//                                                              V hides behind the
//                                                              reduce + sync)
//   partial = q_val * k_j
//   reduce_smem[j & 1][warp_id] = warp_reduce_sum(partial)    (lane 0 of each warp)
//   __syncthreads()
//   s_j     = scale · warp_reduce_sum(reduce_smem[j & 1][lane_id < n_warps ? lane_id : 0])
//   ── per-thread softmax recurrence ──
//   m_new   = max(m, s_j)
//   alpha   = exp(m - m_new)
//   p_j     = exp(s_j - m_new)
//   o_acc   = o_acc * alpha + p_j * v_j
//   l       = l     * alpha + p_j
//   m       = m_new
// At end:  out[tid] = o_acc / l
//
// fp16 in/out, fp32 accumulation.
//
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive, two-pass softmax
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions (single-sync block reduce)              <-- here
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

__global__ void decode_attention_kernel_v2(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale) {

    // Double-buffered per-warp partials. Slot j&1 is written during iter j and
    // read at the end of iter j. Iter j+1 writes the other slot, so it never
    // collides with iter j's read.
    __shared__ float reduce_smem[2][32];

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

    const float q_val = __half2float(q_ptr[tid]);

    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state = 0.0f;

    for (int j = 0; j < seqlen_kv; ++j) {
        const int buf = j & 1;

        // ---- 1. Issue K + V loads up front. V hides behind the reduce + sync.
        const half k_j = k_base[j * head_dim + tid];
        const half v_j = v_base[j * head_dim + tid];

        // ---- 2. Per-warp partial → cross-warp slot ----
        float partial = q_val * __half2float(k_j);
        partial = warp_reduce_sum(partial);
        if (lane_id == 0) reduce_smem[buf][warp_id] = partial;
        __syncthreads();

        // ---- 3. Every warp redundantly does the final cross-warp reduce.
        // After warp_reduce_sum, all 32 lanes of every warp hold the same
        // value, so s_j is available everywhere without a broadcast slot.
        float r = (lane_id < n_warps) ? reduce_smem[buf][lane_id] : 0.0f;
        r = warp_reduce_sum(r);
        const float s_j = r * softmax_scale;

        // ---- 4. Online softmax + output accumulator update ----
        const float m_new = fmaxf(m_state, s_j);
        const float alpha = __expf(m_state - m_new);
        const float p_j   = __expf(s_j     - m_new);
        o_state = o_state * alpha + p_j * __half2float(v_j);
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
    decode_attention_kernel_v2<<<grid, block, 0, stream>>>(
        q, k, v, out, batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale);
    CUDA_CHECK_LAST();
}
