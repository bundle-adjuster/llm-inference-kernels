#pragma once
#include <cstdio>
#include <cstdlib>
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
