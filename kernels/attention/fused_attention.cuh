#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Fused decode attention — one query token per (batch, head) against a KV
// cache of length seqlen_kv. GQA: query heads share kv heads.
// Design + optimization roadmap: docs/01-fused-attention.md
//
// Tensor layouts (row-major, contiguous):
//   q   : [batch, n_heads,    head_dim]
//   k   : [batch, n_kv_heads, seqlen_kv, head_dim]
//   v   : [batch, n_kv_heads, seqlen_kv, head_dim]
//   out : [batch, n_heads,    head_dim]
//
// softmax_scale is typically 1 / sqrt(head_dim).
void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream);
