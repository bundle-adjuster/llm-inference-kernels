#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// Decode attention — naive v0 baseline.
// Layout & contract: kernels/attention/fused_attention.cuh
// Algorithm (per design doc §Design — docs/01-fused-attention.md):
//   one thread block per (batch, head); blockDim.x == head_dim
//   phase 1 (scores):   s[j] = scale · dot(q, k[b, kv(h), j, :])  for j in [0, seqlen_kv)
//   phase 2 (softmax):  block-reduce max; exponentiate (s − max); block-reduce sum
//   phase 3 (output):   o[d] = (1/sum) · Σ_j exp_s[j] · v[b, kv(h), j, d]
// fp16 in/out, fp32 accumulation. Score buffer lives in dynamic shared memory.
//
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive, two-pass softmax                                      <-- here
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions for dot-products and m/l (refinement)
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

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
    }
    return v;
}

}  // namespace

__global__ void decode_attention_kernel_v0(
    const half* __restrict__ q, const half* __restrict__ k,
    const half* __restrict__ v, half* __restrict__ out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale) {

    extern __shared__ float s_smem[];   // [seqlen_kv]  scores, then exp(score − max)
    __shared__ float reduce_smem[32];   // one slot per warp for cross-warp reductions

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

    // Each thread owns one q lane in a register (blockDim.x == head_dim).
    const float q_val = __half2float(q_ptr[tid]);

    // ---- Phase 1: scores ----
    // For each j, compute scale · dot(q, k[j, :]) via a block-wide reduction.
    for (int j = 0; j < seqlen_kv; ++j) {
        float partial = q_val * __half2float(k_base[j * head_dim + tid]);
        partial = warp_reduce_sum(partial);
        if (lane_id == 0) reduce_smem[warp_id] = partial;
        __syncthreads();
        if (warp_id == 0) {
            float r = (lane_id < n_warps) ? reduce_smem[lane_id] : 0.0f;
            r = warp_reduce_sum(r);
            if (lane_id == 0) s_smem[j] = r * softmax_scale;
        }
        __syncthreads();
    }

    // ---- Phase 2: softmax over s_smem ----
    // 2a. block max
    float local_max = -INFINITY;
    for (int j = tid; j < seqlen_kv; j += bdim) {
        local_max = fmaxf(local_max, s_smem[j]);
    }
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) reduce_smem[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float r = (lane_id < n_warps) ? reduce_smem[lane_id] : -INFINITY;
        r = warp_reduce_max(r);
        if (lane_id == 0) reduce_smem[0] = r;
    }
    __syncthreads();
    const float block_max = reduce_smem[0];
    __syncthreads();  // about to overwrite reduce_smem[0..n_warps)

    // 2b. exponentiate (store back into s_smem); accumulate sum
    float local_sum = 0.0f;
    for (int j = tid; j < seqlen_kv; j += bdim) {
        const float e = __expf(s_smem[j] - block_max);
        s_smem[j] = e;
        local_sum += e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) reduce_smem[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float r = (lane_id < n_warps) ? reduce_smem[lane_id] : 0.0f;
        r = warp_reduce_sum(r);
        if (lane_id == 0) reduce_smem[0] = r;
    }
    __syncthreads();
    const float inv_sum = 1.0f / reduce_smem[0];

    // ---- Phase 3: output ----
    // Each thread accumulates its own output lane d = tid across all kv positions.
    float o = 0.0f;
    for (int j = 0; j < seqlen_kv; ++j) {
        o += s_smem[j] * __half2float(v_base[j * head_dim + tid]);
    }
    o_ptr[tid] = __float2half(o * inv_sum);
}

void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    dim3 block(head_dim);
    dim3 grid(batch, n_heads);
    const size_t smem_bytes = static_cast<size_t>(seqlen_kv) * sizeof(float);
    decode_attention_kernel_v0<<<grid, block, smem_bytes, stream>>>(
        q, k, v, out, batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale);
    CUDA_CHECK_LAST();
}
