#include "kv_cache/kv_compress.cuh"
#include "common/cuda_utils.cuh"

// INT8 per-token symmetric quantization for KV-cache.
// Layout & contract: kernels/kv_cache/kv_compress.cuh
//
// One block per (batch, kv_head, token) — grid is (batch, n_kv_heads, seqlen).
// Block is one warp (32 threads); each thread vectorizes 4 fp16 lanes (VEC=4,
// head_dim/32). Within a block:
//
//   1. Load 4 fp16 values per thread (LDG.E.64 via load_half4_as_float4).
//   2. Per-thread absmax over those 4 lanes.
//   3. warp_reduce_max → block-wide absmax (single warp, no shmem).
//   4. scale = max / 127, clamped to >= 1e-8.
//   5. Quantize the 4 lanes: round(x / scale), clamp [-127, 127] → int8.
//   6. Vectorized store of 4 int8 as one int32 (4 bytes per thread).
//   7. Thread 0 writes the fp16 scale.

namespace {

constexpr int VEC = 4;  // halves per thread per load (8 B); also head_dim/32

}  // namespace

__global__ void quantize_per_token_kernel(
    const half* __restrict__ x, int8_t* __restrict__ x_q,
    half* __restrict__ scales,
    int batch, int n_kv_heads, int seqlen, int head_dim) {

    const int batch_idx   = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int token_idx   = blockIdx.z;
    const int tid         = threadIdx.x;  // 0..31; also lane_id

    // Row offset = position of this (batch, kv_head, token) in the flat layout.
    const int row = (batch_idx * n_kv_heads + kv_head_idx) * seqlen + token_idx;

    const half*  x_ptr  = x   + row * head_dim + tid * VEC;
    int8_t*      xq_ptr = x_q + row * head_dim + tid * VEC;

    // 4 fp16 → float4 (one LDG.E.64).
    const float4 vals = load_half4_as_float4(x_ptr);

    // Per-thread absmax over 4 lanes, then warp reduce.
    float local_max = fmaxf(fmaxf(fabsf(vals.x), fabsf(vals.y)),
                            fmaxf(fabsf(vals.z), fabsf(vals.w)));
    const float max_val = warp_reduce_max(local_max);

    // Symmetric scale; tiny epsilon guards against the all-zeros case.
    const float scale     = fmaxf(max_val / 127.0f, 1e-8f);
    const float inv_scale = 1.0f / scale;

    // Quantize each lane: round-to-nearest-even, clamp to [-127, 127].
    int8_t q[VEC];
    q[0] = static_cast<int8_t>(__float2int_rn(fmaxf(-127.0f, fminf(127.0f, vals.x * inv_scale))));
    q[1] = static_cast<int8_t>(__float2int_rn(fmaxf(-127.0f, fminf(127.0f, vals.y * inv_scale))));
    q[2] = static_cast<int8_t>(__float2int_rn(fmaxf(-127.0f, fminf(127.0f, vals.z * inv_scale))));
    q[3] = static_cast<int8_t>(__float2int_rn(fmaxf(-127.0f, fminf(127.0f, vals.w * inv_scale))));

    // 4 int8 as one int32 (4-byte STG).
    *reinterpret_cast<int32_t*>(xq_ptr) = *reinterpret_cast<const int32_t*>(q);

    // One scale per token: lane 0 writes.
    if (tid == 0) {
        scales[row] = __float2half(scale);
    }
}

void launch_quantize_per_token(
    const half* x, int8_t* x_q, half* scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    cudaStream_t stream) {
    // head_dim==128 (= VEC * 32) is the only supported shape — see header.
    dim3 grid(batch, n_kv_heads, seqlen);
    dim3 block(32);
    quantize_per_token_kernel<<<grid, block, 0, stream>>>(
        x, x_q, scales, batch, n_kv_heads, seqlen, head_dim);
    CUDA_CHECK_LAST();
}
