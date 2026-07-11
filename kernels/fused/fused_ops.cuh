#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Fused decode-path elementwise ops (Phase 10). See fused_ops.cu.

// RMSNorm over the last dim: out = weight * x / sqrt(mean(x^2) + eps).
// x, out: [M, H] fp16 (row-major); weight: [H] fp16.
void launch_rmsnorm(const half* x, const half* weight, half* out,
                    int M, int H, float eps, cudaStream_t stream);

// SwiGLU activation: out = silu(gate) * up, elementwise over `total` elements.
void launch_silu_mul(const half* gate, const half* up, half* out,
                     int total, cudaStream_t stream);

// RoPE: out = x*cos + rotate_half(x)*sin. x/out: [B, H, S, D]; cos/sin: [B, S, D].
void launch_rope(const half* x, const half* cos, const half* sin, half* out,
                 int B, int H, int S, int D, cudaStream_t stream);
