#pragma once
#include <cstdint>
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


// Variant of the decode kernel reading INT8 K/V with per-token fp16 scales,
// fusing dequantization into the inner loop. See docs/02-kv-cache-compression.md.
//
// Tensor layouts:
//   q        : [batch, n_heads,    head_dim]                  fp16
//   k_q      : [batch, n_kv_heads, seqlen_kv, head_dim]       int8
//   k_scale  : [batch, n_kv_heads, seqlen_kv]                 fp16 (per-token)
//   v_q      : [batch, n_kv_heads, seqlen_kv, head_dim]       int8
//   v_scale  : [batch, n_kv_heads, seqlen_kv]                 fp16 (per-token)
//   out      : [batch, n_heads,    head_dim]                  fp16
void launch_decode_attention_int8(
    const half* q,
    const int8_t* k_q, const half* k_scale,
    const int8_t* v_q, const half* v_scale,
    half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream);
