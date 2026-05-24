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


// ---- INT4 K (per-channel groupwise) ----
//
// One block per (batch, kv_head, group). Block is one warp (32 threads).
// Each thread owns 4 channels (head_dim/32). Two passes:
//   pass 1 — iterate t in [group_start, group_end), update per-thread
//            per-channel running absmax (4 floats per thread).
//   pass 2 — quantize each (t, channel) using per-channel scales, pack
//            4 nibbles into 2 bytes, single uint16 store per token per
//            thread.
//
// No cross-thread reduction: each thread owns its 4 channels' max
// independently. Per-channel scales never cross thread boundaries.

constexpr int QMAX_INT4 = 7;

__global__ void quantize_k_per_channel_groupwise_int4_kernel(
    const half* __restrict__ x, int8_t* __restrict__ x_q,
    half* __restrict__ scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    int group_size, int n_groups) {

    const int batch_idx   = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int group_idx   = blockIdx.z;
    const int tid         = threadIdx.x;

    const int t_start = group_idx * group_size;
    const int t_end   = min(t_start + group_size, seqlen);

    const int x_row_base = (batch_idx * n_kv_heads + kv_head_idx) * seqlen;

    // ---- Pass 1: per-thread per-channel absmax over the group ----
    float local_max[VEC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int t = t_start; t < t_end; ++t) {
        const float4 v = load_half4_as_float4(
            x + (x_row_base + t) * head_dim + tid * VEC);
        local_max[0] = fmaxf(local_max[0], fabsf(v.x));
        local_max[1] = fmaxf(local_max[1], fabsf(v.y));
        local_max[2] = fmaxf(local_max[2], fabsf(v.z));
        local_max[3] = fmaxf(local_max[3], fabsf(v.w));
    }

    float scale[VEC];
    float inv_scale[VEC];
    #pragma unroll
    for (int d = 0; d < VEC; ++d) {
        scale[d]     = fmaxf(local_max[d] / static_cast<float>(QMAX_INT4), 1e-8f);
        inv_scale[d] = 1.0f / scale[d];
    }

    // Write the 4 scales for this thread's channels (one fp16 each → 8 B
    // packed as float4-to-half4).
    const int s_off = ((batch_idx * n_kv_heads + kv_head_idx) * n_groups
                       + group_idx) * head_dim + tid * VEC;
    store_float4_as_half4(scales + s_off,
                          make_float4(scale[0], scale[1], scale[2], scale[3]));

    // ---- Pass 2: quantize each token, pack 4 nibbles into 2 bytes ----
    const int q_packed_d = head_dim / 2;  // bytes per token in packed layout
    for (int t = t_start; t < t_end; ++t) {
        const float4 v = load_half4_as_float4(
            x + (x_row_base + t) * head_dim + tid * VEC);
        int q[VEC];
        q[0] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, v.x * inv_scale[0])));
        q[1] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, v.y * inv_scale[1])));
        q[2] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, v.z * inv_scale[2])));
        q[3] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, v.w * inv_scale[3])));

        // Pack: byte0 = [q0_lo_nibble, q1_hi_nibble]; byte1 = [q2_lo, q3_hi].
        const uint16_t packed = static_cast<uint16_t>(
            (q[0] & 0xF) | ((q[1] & 0xF) << 4) |
            ((q[2] & 0xF) << 8) | ((q[3] & 0xF) << 12));
        const int q_off = (x_row_base + t) * q_packed_d + tid * 2;
        *reinterpret_cast<uint16_t*>(x_q + q_off) = packed;
    }
}

void launch_quantize_k_per_channel_groupwise_int4(
    const half* x, int8_t* x_q, half* scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    int group_size, int n_groups,
    cudaStream_t stream) {
    dim3 grid(batch, n_kv_heads, n_groups);
    dim3 block(32);
    quantize_k_per_channel_groupwise_int4_kernel<<<grid, block, 0, stream>>>(
        x, x_q, scales, batch, n_kv_heads, seqlen, head_dim,
        group_size, n_groups);
    CUDA_CHECK_LAST();
}


// ---- INT4 V (per-token) ----
//
// Same structure as the INT8 per-token kernel but qmax=7 and the output
// is packed: 4 nibbles per thread → one uint16 store (2 bytes).

__global__ void quantize_v_per_token_int4_kernel(
    const half* __restrict__ x, int8_t* __restrict__ x_q,
    half* __restrict__ scales,
    int batch, int n_kv_heads, int seqlen, int head_dim) {

    const int batch_idx   = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int token_idx   = blockIdx.z;
    const int tid         = threadIdx.x;

    const int row = (batch_idx * n_kv_heads + kv_head_idx) * seqlen + token_idx;

    const half* x_ptr = x + row * head_dim + tid * VEC;
    // Packed output: head_dim/2 bytes per token, 2 bytes per thread.
    int8_t* xq_ptr = x_q + row * (head_dim / 2) + tid * 2;

    const float4 vals = load_half4_as_float4(x_ptr);

    float local_max = fmaxf(fmaxf(fabsf(vals.x), fabsf(vals.y)),
                            fmaxf(fabsf(vals.z), fabsf(vals.w)));
    const float max_val = warp_reduce_max(local_max);

    const float scale     = fmaxf(max_val / static_cast<float>(QMAX_INT4), 1e-8f);
    const float inv_scale = 1.0f / scale;

    int q[VEC];
    q[0] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, vals.x * inv_scale)));
    q[1] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, vals.y * inv_scale)));
    q[2] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, vals.z * inv_scale)));
    q[3] = __float2int_rn(fmaxf(-7.0f, fminf(7.0f, vals.w * inv_scale)));

    const uint16_t packed = static_cast<uint16_t>(
        (q[0] & 0xF) | ((q[1] & 0xF) << 4) |
        ((q[2] & 0xF) << 8) | ((q[3] & 0xF) << 12));
    *reinterpret_cast<uint16_t*>(xq_ptr) = packed;

    if (tid == 0) {
        scales[row] = __float2half(scale);
    }
}

void launch_quantize_v_per_token_int4(
    const half* x, int8_t* x_q, half* scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    cudaStream_t stream) {
    dim3 grid(batch, n_kv_heads, seqlen);
    dim3 block(32);
    quantize_v_per_token_int4_kernel<<<grid, block, 0, stream>>>(
        x, x_q, scales, batch, n_kv_heads, seqlen, head_dim);
    CUDA_CHECK_LAST();
}
