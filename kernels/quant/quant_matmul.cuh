#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// W4A16 GEMM — out[M,N] = act[M,K] @ dequant(weight[K,N]).
// Design: docs/03-quantized-matmul.md
//
//   act           : [M, K]  FP16 activations
//   weight_packed : INT4 weights, 8 values packed per uint32_t, [K, N/8]
//   scales        : group-wise FP16 scales along K (group_size, e.g. 128)
//   out           : [M, N]  FP16
void launch_w4a16_gemm(
    const half* act, const uint32_t* weight_packed, const half* scales,
    half* out, int M, int N, int K, int group_size, cudaStream_t stream);
