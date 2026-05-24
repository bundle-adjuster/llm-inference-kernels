#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention reading INT4 KIVI-quantized K/V with fused dequantization.
// Layout & contract: kernels/attention/fused_attention.cuh
// Design: docs/02-kv-cache-compression.md
//
// KIVI:
//   K — per-channel groupwise scales: one fp16 scale per (group, channel),
//       where group_size tokens share scales along the seqlen axis.
//   V — per-token scales: one fp16 scale per token (shared across head_dim).
//
// Built on v3 + the INT8 path's scale-folding tricks, with two adjustments
// forced by the per-channel K structure:
//
//   1. Per-channel K scales can't be folded out of the dot product like
//      INT8's per-token scale was (which was a *scalar* and pulled out of
//      Σ_d). But they're constant within a group, so we pre-scale q ONCE
//      per group:  q_scaled[d] = q_v[d] * k_scale[group, d].  The inner
//      loop's dot product is then 4 multiplies on int values — no per-iter
//      dequant cost from K.
//
//   2. The V scale is still per-token, so the INT8 trick still works:
//      fold v_scale into the FMA coefficient (p_j_scaled = p_j * v_scale).
//
// Per kv position j inside group g:
//   k_int  = load_int4x4_as_float4(K_q[j, tid*VEC/2 ..])    (one uint16 load)
//   v_int  = load_int4x4_as_float4(V_q[j, ...])
//   v_s    = fp16->float of V_scale[j]
//   partial= Σ_d q_scaled[d] · k_int[d]
//   s_j    = softmax_scale · warp_reduce_sum(partial)
//   ── online softmax ──
//   m_new  = max(m, s_j)
//   alpha  = exp(m - m_new)
//   p_j    = exp(s_j - m_new)
//   p_j_v  = p_j · v_s
//   o_v[d] = o_v[d] · alpha + p_j_v · v_int[d]
// At end: out[tid*VEC ..] = o_v / l         (vectorized fp16 store)

namespace {
constexpr int VEC = 4;
}  // namespace

__global__ void decode_attention_int4_kernel(
    const half* __restrict__ q,
    const int8_t* __restrict__ k_q, const half* __restrict__ k_scale,
    const int8_t* __restrict__ v_q, const half* __restrict__ v_scale,
    half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int group_size, int n_groups,
    float softmax_scale) {

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const int packed_d = head_dim / 2;

    const half*   q_ptr    = q + (batch_idx * n_heads    + head_idx)    * head_dim + tid * VEC;
    const int8_t* k_q_base = k_q + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * packed_d;
    const half*   k_s_base = k_scale + (batch_idx * n_kv_heads + kv_head_idx) * n_groups * head_dim;
    const int8_t* v_q_base = v_q + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * packed_d;
    const half*   v_s_base = v_scale + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv;
    half*         o_ptr    = out + (batch_idx * n_heads + head_idx)     * head_dim + tid * VEC;

    const float4 q_v = load_half4_as_float4(q_ptr);

    float4 o_v     = {0.0f, 0.0f, 0.0f, 0.0f};
    float  m_state = -INFINITY;
    float  l_state = 0.0f;

    // q_scaled is recomputed once per group: q_scaled[d] = q_v[d] · k_scale[g, d].
    float4 q_scaled;

    for (int g = 0; g < n_groups; ++g) {
        // Load 4 K scales for this thread's channels (one fp16 each).
        const float4 k_s = load_half4_as_float4(k_s_base + g * head_dim + tid * VEC);
        q_scaled.x = q_v.x * k_s.x;
        q_scaled.y = q_v.y * k_s.y;
        q_scaled.z = q_v.z * k_s.z;
        q_scaled.w = q_v.w * k_s.w;

        const int t_start = g * group_size;
        const int t_end   = min(t_start + group_size, seqlen_kv);

        for (int t = t_start; t < t_end; ++t) {
            // 2-byte load per thread (4 nibbles → float4) for K and V.
            const float4 k_int = load_int4x4_as_float4(k_q_base + t * packed_d + tid * 2);
            const float4 v_int = load_int4x4_as_float4(v_q_base + t * packed_d + tid * 2);
            const float  v_s   = __half2float(v_s_base[t]);

            // Dot product on int values; K-side scale is already in q_scaled.
            float partial = q_scaled.x * k_int.x + q_scaled.y * k_int.y
                          + q_scaled.z * k_int.z + q_scaled.w * k_int.w;
            partial = warp_reduce_sum(partial);
            const float s_j = partial * softmax_scale;

            const float m_new = fmaxf(m_state, s_j);
            const float alpha = __expf(m_state - m_new);
            const float p_j   = __expf(s_j     - m_new);
            const float p_j_v = p_j * v_s;

            o_v.x = o_v.x * alpha + p_j_v * v_int.x;
            o_v.y = o_v.y * alpha + p_j_v * v_int.y;
            o_v.z = o_v.z * alpha + p_j_v * v_int.z;
            o_v.w = o_v.w * alpha + p_j_v * v_int.w;

            l_state = l_state * alpha + p_j;
            m_state = m_new;
        }
    }

    const float inv_l = 1.0f / l_state;
    o_v.x *= inv_l;  o_v.y *= inv_l;  o_v.z *= inv_l;  o_v.w *= inv_l;
    store_float4_as_half4(o_ptr, o_v);
}

void launch_decode_attention_int4(
    const half* q,
    const int8_t* k_q, const half* k_scale,
    const int8_t* v_q, const half* v_scale,
    half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int group_size, int n_groups,
    float softmax_scale, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch, n_heads);
    decode_attention_int4_kernel<<<grid, block, 0, stream>>>(
        q, k_q, k_scale, v_q, v_scale, out,
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
        group_size, n_groups, softmax_scale);
    CUDA_CHECK_LAST();
}
