#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// KV-cache quantization — FP16 -> INT8 with per-token scales.
// Design: docs/02-kv-cache-compression.md
//
// Tensor layouts (row-major, contiguous):
//   x       : [batch, n_kv_heads, seqlen, head_dim]  fp16
//   x_q     : [batch, n_kv_heads, seqlen, head_dim]  int8 (per-token symmetric)
//   scales  : [batch, n_kv_heads, seqlen]            fp16 (one scale per token)
//
// Symmetric: scale = max(|x| over head_dim) / 127, clamped to >= 1e-8.
// Quantize: q = round(x / scale), clamped to [-127, 127].
// Reference oracle in PyTorch: reference/kv_cache_ref.py quantize_per_token.
//
// Assumes head_dim == 128 (single-warp block with vec=4 per thread).
void launch_quantize_per_token(
    const half* x, int8_t* x_q, half* scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    cudaStream_t stream);


// INT4 KIVI quantization — K per-channel groupwise, V per-token.
// Each int8 byte holds two signed 4-bit values in [-7, 7]:
//   byte = (q_lo & 0xF) | ((q_hi & 0xF) << 4)
//   where q_lo is at the even channel index and q_hi at the odd.
//
// K (per-channel groupwise):
//   x       : [batch, n_kv_heads, seqlen,     head_dim    ]  fp16
//   x_q     : [batch, n_kv_heads, seqlen,     head_dim/2  ]  int8 (packed)
//   scales  : [batch, n_kv_heads, n_groups,   head_dim    ]  fp16
//     where n_groups = ceil(seqlen / group_size).
void launch_quantize_k_per_channel_groupwise_int4(
    const half* x, int8_t* x_q, half* scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    int group_size, int n_groups,
    cudaStream_t stream);

// V (per-token):
//   x       : [batch, n_kv_heads, seqlen, head_dim    ]  fp16
//   x_q     : [batch, n_kv_heads, seqlen, head_dim/2  ]  int8 (packed)
//   scales  : [batch, n_kv_heads, seqlen]               fp16
void launch_quantize_v_per_token_int4(
    const half* x, int8_t* x_q, half* scales,
    int batch, int n_kv_heads, int seqlen, int head_dim,
    cudaStream_t stream);
