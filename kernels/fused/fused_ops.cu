#include "fused/fused_ops.cuh"
#include "common/cuda_utils.cuh"

// Fused elementwise ops for the decode path (Phase 10). transformers runs each of
// these as several separate kernels (RMSNorm = pow+mean+rsqrt+mul+mul; SwiGLU =
// silu then mul); a serving engine fuses them. At batch=16 the projection GEMMs
// dominate, but the unfused elementwise is a measurable slice of what still
// separates the fp16 stack from vLLM. Each op here is one kernel, one pass.

namespace {
constexpr int THREADS = 256;

// Block reduce over `s[32]` scratch; returns the sum to every thread.
__device__ __forceinline__ float block_reduce_sum(float v, float* s) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? s[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    return s[0];
}
}  // namespace

// RMSNorm: out[r] = weight * x[r] / sqrt(mean(x[r]^2) + eps). One block per row,
// fp32 reduction (matches transformers' fp32 variance). x/out: [M, H] fp16.
__global__ void rmsnorm_kernel(
    const half* __restrict__ x, const half* __restrict__ weight,
    half* __restrict__ out, int M, int H, float eps) {

    const int row = blockIdx.x;
    if (row >= M) return;
    const half* xr = x + static_cast<long>(row) * H;
    half* outr     = out + static_cast<long>(row) * H;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        ss += v * v;
    }
    __shared__ float s[32];
    ss = block_reduce_sum(ss, s);
    const float inv = rsqrtf(ss / H + eps);

    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        outr[i] = __float2half(__half2float(xr[i]) * inv * __half2float(weight[i]));
    }
}

// SwiGLU: out = silu(gate) * up, elementwise. silu(x) = x * sigmoid(x).
__global__ void silu_mul_kernel(
    const half* __restrict__ gate, const half* __restrict__ up,
    half* __restrict__ out, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float g = __half2float(gate[i]);
    const float silu = g / (1.0f + __expf(-g));
    out[i] = __float2half(silu * __half2float(up[i]));
}

// RoPE: out = x*cos + rotate_half(x)*sin, with rotate_half(x) = [-x[D/2:], x[:D/2]].
// x/out: [B, H, S, D] fp16; cos/sin: [B, S, D] fp16 (Llama's duplicated-freq layout).
// One thread per (b, h, s, d). Fuses transformers' mul/cat/mul/add into one pass.
__global__ void rope_kernel(
    const half* __restrict__ x, const half* __restrict__ cos,
    const half* __restrict__ sin, half* __restrict__ out,
    int B, int H, int S, int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * S * D) return;
    const int d = idx % D;
    const int s = (idx / D) % S;
    const int b = idx / (static_cast<long>(D) * S * H);
    const int half_d = D >> 1;

    const long row = idx - d;                        // start of this (b,h,s) vector
    const float rot = (d < half_d)
        ? -__half2float(x[row + d + half_d])
        :  __half2float(x[row + d - half_d]);
    const int cs = (b * S + s) * D + d;
    out[idx] = __float2half(__half2float(x[idx]) * __half2float(cos[cs])
                            + rot * __half2float(sin[cs]));
}

void launch_rope(const half* x, const half* cos, const half* sin, half* out,
                 int B, int H, int S, int D, cudaStream_t stream) {
    const int total = B * H * S * D;
    rope_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
        x, cos, sin, out, B, H, S, D);
    CUDA_CHECK_LAST();
}

void launch_rmsnorm(const half* x, const half* weight, half* out,
                    int M, int H, float eps, cudaStream_t stream) {
    rmsnorm_kernel<<<M, THREADS, 0, stream>>>(x, weight, out, M, H, eps);
    CUDA_CHECK_LAST();
}

void launch_silu_mul(const half* gate, const half* up, half* out,
                     int total, cudaStream_t stream) {
    const int blocks = (total + THREADS - 1) / THREADS;
    silu_mul_kernel<<<blocks, THREADS, 0, stream>>>(gate, up, out, total);
    CUDA_CHECK_LAST();
}
