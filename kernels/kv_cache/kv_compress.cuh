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
