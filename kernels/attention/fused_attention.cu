#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// TODO(Phase 1) — implement the decode attention kernel.
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive: one block per (batch, head), two-pass softmax  <-- start here
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions for dot-products and m/l
//   v3  vectorized 128-bit coalesced KV loads
//   v4  split-K over the KV sequence + combine (FlashDecoding)
//   v5  cp.async double-buffering of KV tiles

__device__ half exp(half x) {
    return __expf(x);
}
__global__ void decode_attention_kernel_v0(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    // TODO(Phase 1b): implement the v0 kernel here.
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int thread_idx = threadIdx.x;

    // Get the pointers for the current batch and head
    const half* q_ptr = q + batch_idx * n_heads * head_dim + head_idx * head_dim;
    const half* k_ptr = k + batch_idx * n_kv_heads * seqlen_kv * head_dim + head_idx * n_kv_heads * seqlen_kv * head_dim;
    const half* v_ptr = v + batch_idx * n_kv_heads * seqlen_kv * head_dim + head_idx * n_kv_heads * seqlen_kv * head_dim;
    half* out_ptr = out + batch_idx * n_heads * head_dim + head_idx * head_dim;

    // Compute the dot product between the query and key
    half dot_product = 0;
    for (int i = 0; i < head_dim; i++) {
        dot_product += q_ptr[i] * k_ptr[i];
    }

    // Compute the softmax of the dot product
    half softmax_value = __expf(static_cast<float>(dot_product) * softmax_scale);

    // Compute the output
    out_ptr[thread_idx] = softmax_value * v_ptr[thread_idx];
}
void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    // TODO(Phase 1b): launch the v0 kernel here.
    dim3 block(head_dim);
    dim3 grid(batch, n_heads);
    decode_attention_kernel_v0<<<grid, block, 0, stream>>>(q, k, v, out, batch, n_heads, n_kv_heads, seqlen_kv, head_dim, softmax_scale, stream);
}
