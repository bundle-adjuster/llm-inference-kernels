#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention — v4, FlashDecoding (split-K).
// Layout & contract: kernels/attention/fused_attention.cuh
//
// Why split-K: v3 launched batch × n_heads = 256 blocks at 16 blocks/SM, so
// only ~16 of the 4090's 128 SMs were busy. For decode workloads with small
// batches the (batch × heads) grid underfills the GPU. Split-K partitions
// each (batch, head) across K_SPLIT chunks of the KV sequence, multiplying
// the grid size and exposing enough parallelism to keep all SMs busy.
//
// Two kernels:
//
//   Stage 1 (decode_attention_split_kernel): grid = (batch, n_heads, K_SPLIT),
//     block = 32. Each block runs the same body as v3 — vectorized KV loads,
//     online softmax — but only over j in [split_idx * chunk, (split+1) * chunk).
//     Writes the unnormalized o_acc[head_dim] and (m, l) per split to scratch.
//
//   Stage 2 (decode_attention_combine_kernel): grid = (batch, n_heads),
//     block = 32. Reads K_SPLIT partial (m_s, l_s, o_s[d]) tuples and merges
//     them via the FlashAttention online-softmax combine:
//
//       m_final    = max_s m_s
//       l_final    = Σ_s l_s · exp(m_s − m_final)
//       o_final[d] = (Σ_s o_s[d] · exp(m_s − m_final)) / l_final
//
// Scratch: ~528 KB for our reference workload. Stream-ordered alloc/free via
// cudaMallocAsync / cudaFreeAsync — no API change to launch_decode_attention.
//
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive, two-pass softmax
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions (single-sync block reduce)
//   v3  vectorized KV loads (single-warp block, 64-bit per thread)
//   v4  split-K over the KV sequence + combine (FlashDecoding)        <-- here
//   v5  cp.async double-buffering of KV tiles

namespace {

constexpr int VEC     = 4;   // halves per thread per vec load
constexpr int K_SPLIT = 8;   // split factor along KV sequence

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
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

// ---- Stage 1: partial decode over [j_start, j_end) ----
__global__ void decode_attention_split_kernel(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v,
    float* __restrict__ m_scratch, float* __restrict__ l_scratch,
    half*  __restrict__ o_scratch,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int chunk, float softmax_scale) {

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid       = threadIdx.x;

    const int j_start = split_idx * chunk;
    const int j_end   = min(j_start + chunk, seqlen_kv);

    const int ml_idx = (batch_idx * n_heads + head_idx) * K_SPLIT + split_idx;
    const int o_idx  = ml_idx * head_dim + tid * VEC;

    // Empty range (only when seqlen_kv < K_SPLIT). Sentinel: (-INF, 0, 0).
    if (j_start >= seqlen_kv) {
        if (tid == 0) {
            m_scratch[ml_idx] = -INFINITY;
            l_scratch[ml_idx] = 0.0f;
        }
        store_float4_as_half4(o_scratch + o_idx, make_float4(0.f, 0.f, 0.f, 0.f));
        return;
    }

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);

    const half* q_ptr  = q + (batch_idx * n_heads    + head_idx)    * head_dim + tid * VEC;
    const half* k_base = k + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;
    const half* v_base = v + (batch_idx * n_kv_heads + kv_head_idx) * seqlen_kv * head_dim;

    const float4 q_v = load_half4_as_float4(q_ptr);

    float4 o_v     = {0.0f, 0.0f, 0.0f, 0.0f};
    float  m_state = -INFINITY;
    float  l_state = 0.0f;

    for (int j = j_start; j < j_end; ++j) {
        const float4 k_v = load_half4_as_float4(k_base + j * head_dim + tid * VEC);
        const float4 v_v = load_half4_as_float4(v_base + j * head_dim + tid * VEC);

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

    // Write (m, l) once per block; o (unnormalized) once per thread.
    if (tid == 0) {
        m_scratch[ml_idx] = m_state;
        l_scratch[ml_idx] = l_state;
    }
    store_float4_as_half4(o_scratch + o_idx, o_v);
}

// ---- Stage 2: merge K_SPLIT partials into the final output ----
__global__ void decode_attention_combine_kernel(
    const float* __restrict__ m_scratch, const float* __restrict__ l_scratch,
    const half*  __restrict__ o_scratch, half* __restrict__ out,
    int batch, int n_heads, int head_dim) {

    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;

    const int ml_base = (batch_idx * n_heads + head_idx) * K_SPLIT;

    // 1. m_final = max_s m_s
    float m_local = -INFINITY;
    for (int s = tid; s < K_SPLIT; s += 32) {
        m_local = fmaxf(m_local, m_scratch[ml_base + s]);
    }
    const float m_final = warp_reduce_max(m_local);

    // 2. l_final = Σ_s l_s · exp(m_s − m_final)
    float l_local = 0.0f;
    for (int s = tid; s < K_SPLIT; s += 32) {
        const float m_s = m_scratch[ml_base + s];
        const float l_s = l_scratch[ml_base + s];
        l_local += l_s * __expf(m_s - m_final);
    }
    const float l_final = warp_reduce_sum(l_local);
    const float inv_l   = 1.0f / l_final;

    // 3. o_final[d] = (Σ_s o_s[d] · exp(m_s − m_final)) · inv_l
    //    Each thread accumulates 4 d-lanes.
    float4 o_v = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int s = 0; s < K_SPLIT; ++s) {
        const float scale = __expf(m_scratch[ml_base + s] - m_final);
        const int   o_idx = (ml_base + s) * head_dim + tid * VEC;
        const float4 o_s  = load_half4_as_float4(o_scratch + o_idx);
        o_v.x += o_s.x * scale;
        o_v.y += o_s.y * scale;
        o_v.z += o_s.z * scale;
        o_v.w += o_s.w * scale;
    }
    o_v.x *= inv_l;  o_v.y *= inv_l;  o_v.z *= inv_l;  o_v.w *= inv_l;

    half* o_ptr = out + (batch_idx * n_heads + head_idx) * head_dim + tid * VEC;
    store_float4_as_half4(o_ptr, o_v);
}

void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {

    const int chunk = (seqlen_kv + K_SPLIT - 1) / K_SPLIT;

    // One stream-ordered allocation, partitioned into m / l / o regions.
    // (Three separate cudaMallocAsync calls add per-call overhead; one
    // combined call amortises that.)
    const size_t ml_count    = static_cast<size_t>(batch) * n_heads * K_SPLIT;
    const size_t o_count     = ml_count * head_dim;
    const size_t m_bytes     = ml_count * sizeof(float);
    const size_t l_bytes     = ml_count * sizeof(float);
    const size_t o_bytes     = o_count  * sizeof(half);
    const size_t total_bytes = m_bytes + l_bytes + o_bytes;
    char* scratch = nullptr;
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&scratch), total_bytes, stream));
    float* m_scratch = reinterpret_cast<float*>(scratch);
    float* l_scratch = reinterpret_cast<float*>(scratch + m_bytes);
    half*  o_scratch = reinterpret_cast<half*>(scratch + m_bytes + l_bytes);

    // Stage 1
    dim3 split_grid(batch, n_heads, K_SPLIT);
    dim3 split_block(32);
    decode_attention_split_kernel<<<split_grid, split_block, 0, stream>>>(
        q, k, v, m_scratch, l_scratch, o_scratch,
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
        chunk, softmax_scale);
    CUDA_CHECK_LAST();

    // Stage 2
    dim3 combine_grid(batch, n_heads);
    dim3 combine_block(32);
    decode_attention_combine_kernel<<<combine_grid, combine_block, 0, stream>>>(
        m_scratch, l_scratch, o_scratch, out, batch, n_heads, head_dim);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaFreeAsync(scratch, stream));
}
