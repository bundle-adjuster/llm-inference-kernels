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


// ---- INT4 (packed) unpack helpers ----
//
// Packed convention: one int8 byte holds two signed 4-bit values in [-7, 7].
//   byte = (q_lo & 0xF) | ((q_hi & 0xF) << 4)
//   where q_lo is at the EVEN channel index (d, d+2, ...) and q_hi is at
//   the ODD index (d+1, d+3, ...). The reference oracle (PyTorch
//   reference/kv_cache_ref.py) stores in this same layout for direct test
//   comparison.
//
// Sign-extension: shift the nibble into the high 4 bits, then arithmetic
// right shift back. In CUDA PTX, right shift of a signed int is arithmetic
// (sign-preserving), so `static_cast<int8_t>(b << 4) >> 4` works.

__device__ __forceinline__ float unpack_int4_lo(uint8_t b) {
    const int8_t v = static_cast<int8_t>(b << 4) >> 4;
    return static_cast<float>(v);
}

__device__ __forceinline__ float unpack_int4_hi(uint8_t b) {
    const int8_t v = static_cast<int8_t>(b) >> 4;
    return static_cast<float>(v);
}

// 2 packed bytes (4 nibbles) -> float4. One uint16 load is sufficient on
// 2-byte alignment.
__device__ __forceinline__ float4 load_int4x4_as_float4(const int8_t* ptr) {
    const uint16_t packed = *reinterpret_cast<const uint16_t*>(ptr);
    const uint8_t b0 = static_cast<uint8_t>(packed & 0xFF);
    const uint8_t b1 = static_cast<uint8_t>((packed >> 8) & 0xFF);
    return make_float4(
        unpack_int4_lo(b0),
        unpack_int4_hi(b0),
        unpack_int4_lo(b1),
        unpack_int4_hi(b1));
}
