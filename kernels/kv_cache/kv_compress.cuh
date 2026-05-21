#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// KV-cache quantization — FP16 -> INT8/INT4 with group-wise scales.
// Design: docs/02-kv-cache-compression.md
//
// Called when a new token is appended to the KV cache. The matching
// fused-dequant path lives inside the Track 1 decode attention kernel.
//
//   bits = 8 or 4. Packing/layout finalized in Phase 2.
void launch_kv_quantize(
    const half* kv_fp16, int8_t* kv_quant, half* scales,
    int n_kv_heads, int head_dim, int bits, cudaStream_t stream);
