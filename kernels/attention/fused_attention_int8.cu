#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention reading INT8-quantized K/V with fused per-token dequant.
// Layout & contract: kernels/attention/fused_attention.cuh
// Design: docs/02-kv-cache-compression.md
//
// Built on v3 (single-warp block, vectorized loads). The structural changes
// are minimal:
//   - K and V are int8 with one fp16 scale per token (per-token symmetric
//     quantization). Half the KV bytes vs the fp16 path.
//   - Dequantization is fused into the inner loop; no separate dequant pass,
//     no fp16 KV ever materialised in HBM.
//
// **Scale-folding optimisation**: instead of per-lane dequant (`k = k_int *
// k_scale_j`, 4 multiplies per thread per iter), we observe that the dot
// product is linear:
//
//     dot(q, k) = dot(q, k_int * k_scale_j) = k_scale_j * dot(q, k_int)
//
// so we compute the partial dot product on int values, then multiply by
// k_scale_j once after the warp reduce. Similarly for V: fold v_scale_j
// into p_j once (`p_j_scaled = p_j * v_scale_j`), then the V FMA uses int
// values directly. Saves 8 multiplies per iter vs naive dequant-everything.
//
// Per kv position j:
//   k_int, v_int  = load_int8x4_as_float4(K_q[j, tid*VEC..]), V_q[j, ...]
//   k_s, v_s      = fp16 -> float of per-token scales
//   partial       = Σ_d q_v[d] · k_int[d]
//   s_j           = k_s · softmax_scale · warp_reduce_sum(partial)
//   ── online softmax ──
//   m_new   = max(m, s_j)
//   alpha   = exp(m - m_new)
//   p_j     = exp(s_j - m_new)
//   p_j_v   = p_j · v_s                                       // fold V scale
//   o_v[d]  = o_v[d] · alpha + p_j_v · v_int[d]              // FMA on ints
//   l       = l · alpha + p_j
//   m       = m_new
// At end:  out[tid*VEC ..] = o_v / l                          (vectorized)

namespace {
constexpr int VEC = 4;
}  // namespace

__global__ void decode_attention_int8_kernel(
    const half* __restrict__ q,
    const int8_t* __restrict__ k_q, const half* __restrict__ k_scale,
    const int8_t* __restrict__ v_q, const half* __restrict__ v_scale,
    half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale) {

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;   // 0..31; also lane_id

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const half*   q_ptr    = q + (batch_idx * n_heads    + head_idx)    * head_dim + tid * VEC;
    const int8_t* k_q_base = k_q + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half*   k_s_base = k_scale + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv;
    const int8_t* v_q_base = v_q + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half*   v_s_base = v_scale + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv;
    half*         o_ptr    = out + (batch_idx * n_heads + head_idx)     * head_dim + tid * VEC;

    const float4 q_v = load_half4_as_float4(q_ptr);

    float4 o_v     = {0.0f, 0.0f, 0.0f, 0.0f};
    float  m_state = -INFINITY;
    float  l_state = 0.0f;

    for (int j = 0; j < seqlen_kv; ++j) {
        // Loads: scales (broadcast across warp via L1/L2), int8 K and V (one
        // int32 = 4 lanes per thread each). The V loads issue at the top so
        // their latency hides behind the K reduction + softmax update — same
        // prefetch trick as v3.
        const float k_s   = __half2float(k_s_base[j]);
        const float v_s   = __half2float(v_s_base[j]);
        const float4 k_int = load_int8x4_as_float4(k_q_base + j * head_dim + tid * VEC);
        const float4 v_int = load_int8x4_as_float4(v_q_base + j * head_dim + tid * VEC);

        // Partial dot product on int values; fold k_s in after the warp reduce.
        float partial = q_v.x * k_int.x + q_v.y * k_int.y
                      + q_v.z * k_int.z + q_v.w * k_int.w;
        partial = warp_reduce_sum(partial);
        const float s_j = partial * k_s * softmax_scale;

        // Online softmax recurrence.
        const float m_new   = fmaxf(m_state, s_j);
        const float alpha   = __expf(m_state - m_new);
        const float p_j     = __expf(s_j     - m_new);
        const float p_j_v   = p_j * v_s;   // fold V scale into the FMA coeff.

        // V FMA on int values, scaled once by p_j_v.
        o_v.x = o_v.x * alpha + p_j_v * v_int.x;
        o_v.y = o_v.y * alpha + p_j_v * v_int.y;
        o_v.z = o_v.z * alpha + p_j_v * v_int.z;
        o_v.w = o_v.w * alpha + p_j_v * v_int.w;

        l_state = l_state * alpha + p_j;
        m_state = m_new;
    }

    const float inv_l = 1.0f / l_state;
    o_v.x *= inv_l;  o_v.y *= inv_l;  o_v.z *= inv_l;  o_v.w *= inv_l;
    store_float4_as_half4(o_ptr, o_v);
}

void launch_decode_attention_int8(
    const half* q,
    const int8_t* k_q, const half* k_scale,
    const int8_t* v_q, const half* v_scale,
    half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid(batch, n_heads);
    decode_attention_int8_kernel<<<grid, block, 0, stream>>>(
        q, k_q, k_scale, v_q, v_scale, out,
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale);
    CUDA_CHECK_LAST();
}
