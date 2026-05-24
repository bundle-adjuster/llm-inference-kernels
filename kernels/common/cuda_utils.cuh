#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Shared CUDA helpers. No PyTorch dependency — usable from standalone
// microbenchmarks and from the Torch extension alike.

// Wrap every CUDA runtime API call.
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         cudaGetErrorName(err__), __FILE__, __LINE__,           \
                         cudaGetErrorString(err__));                            \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// Call immediately after a <<<>>> kernel launch.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }


// ---- Warp-level reductions (butterfly via shfl_xor) ----

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
    }
    return v;
}


// ---- Vectorized load/store helpers ----
//
// Each compiles to a single LDG.E.64 / STG.E.64 (8 bytes per thread, 4 halves).
// The pointer must be 8-byte aligned, which is always true for torch tensors
// at index `base + tid * 4` since torch tensors are at least 16-byte aligned.

__device__ __forceinline__ float4 load_half4_as_float4(const half* ptr) {
    const uint2 raw = *reinterpret_cast<const uint2*>(ptr);
    half2 lo, hi;
    *reinterpret_cast<unsigned int*>(&lo) = raw.x;
    *reinterpret_cast<unsigned int*>(&hi) = raw.y;
    const float2 f_lo = __half22float2(lo);
    const float2 f_hi = __half22float2(hi);
    return make_float4(f_lo.x, f_lo.y, f_hi.x, f_hi.y);
}

__device__ __forceinline__ void store_float4_as_half4(half* ptr, const float4 v) {
    const half2 lo = __floats2half2_rn(v.x, v.y);
    const half2 hi = __floats2half2_rn(v.z, v.w);
    uint2 raw;
    raw.x = *reinterpret_cast<const unsigned int*>(&lo);
    raw.y = *reinterpret_cast<const unsigned int*>(&hi);
    *reinterpret_cast<uint2*>(ptr) = raw;
}

// 4 signed bytes (one int32 load) -> float4.
__device__ __forceinline__ float4 load_int8x4_as_float4(const int8_t* ptr) {
    const int32_t packed = *reinterpret_cast<const int32_t*>(ptr);
    // Sign-extend each 8-bit lane to int, then to float.
    return make_float4(
        static_cast<float>(static_cast<int8_t>(packed & 0xFF)),
        static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xFF)),
        static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xFF)),
        static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xFF)));
}
