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


// Decode attention — v6, FlashDecoding split-K. Splits each (batch, head)'s KV
// sequence into `n_splits` chunks processed by independent blocks, then merges
// the partial online-softmax states. Fixes v3's occupancy wall (see
// kernels/attention/fused_attention_splitk.cu and
// docs/05-baseline-correction-journey.md).
//
// Scratch (caller-allocated, fp32): partial_o [batch, n_heads, n_splits,
// head_dim], partial_m / partial_l [batch, n_heads, n_splits]. Use
// decode_attention_n_splits() to size them.
int decode_attention_n_splits(int batch, int n_heads, int seqlen_kv);

void launch_decode_attention_splitk(
    const half* q, const half* k, const half* v, half* out,
    float* partial_o, float* partial_m, float* partial_l,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int n_splits, float softmax_scale, cudaStream_t stream);


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


// INT4 KIVI variant: K per-channel groupwise + V per-token, packed 4-bit storage.
//
// Tensor layouts (packed: one int8 byte = two 4-bit signed values in [-7, 7]):
//   q        : [batch, n_heads,    head_dim   ]                fp16
//   k_q      : [batch, n_kv_heads, seqlen_kv, head_dim/2]      int8 (packed)
//   k_scale  : [batch, n_kv_heads, n_groups,  head_dim ]       fp16
//                (n_groups = ceil(seqlen_kv / group_size))
//   v_q      : [batch, n_kv_heads, seqlen_kv, head_dim/2]      int8 (packed)
//   v_scale  : [batch, n_kv_heads, seqlen_kv]                  fp16 (per-token)
//   out      : [batch, n_heads,    head_dim   ]                fp16
void launch_decode_attention_int4(
    const half* q,
    const int8_t* k_q, const half* k_scale,
    const int8_t* v_q, const half* v_scale,
    half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    int group_size, int n_groups,
    float softmax_scale, cudaStream_t stream);
