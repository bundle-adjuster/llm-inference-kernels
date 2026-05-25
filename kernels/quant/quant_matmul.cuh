#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// W4A16 GEMM — out[M, N] = act[M, K] @ dequant(weight[K, N]).
// Design: docs/03-quantized-matmul.md
//
// Weights are quantized once offline (host-side or via a separate
// quantization pipeline) using symmetric INT4 with groupwise scales
// along K. The kernel reads packed bytes from HBM, dequantizes in
// registers, and FMAs against fp16 activations. No fp16 W ever
// materialises in HBM — that's the bandwidth win for decode.
//
// Tensor layouts (row-major, contiguous):
//
//   act           : [M, K]                 fp16
//   weight_packed : [K/8, N]               uint32 (8 nibbles per word
//                                          along K — bit `i*4..i*4+3`
//                                          stores K position `k_pack*8 + i`)
//   scales        : [n_groups, N]          fp16 where
//                                          n_groups = K / group_size
//   out           : [M, N]                 fp16
//
// Requirements:
//   - K is a multiple of group_size and group_size is a multiple of 8
//     (so K / 8 is the packed dim and K / group_size is the scale dim).
//   - head_dim shapes in Llama 3 8B (K ∈ {4096, 14336}, group_size = 128)
//     satisfy this trivially.
void launch_w4a16_gemm(
    const half* act, const uint32_t* weight_packed, const half* scales,
    half* out, int M, int N, int K, int group_size, cudaStream_t stream);
