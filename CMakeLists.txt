cmake_minimum_required(VERSION 3.24)
project(llm_inference_kernels LANGUAGES CXX CUDA)

# Standalone build for pure-CUDA microbenchmarks (no PyTorch dependency).
# The PyTorch extension is built separately via setup.py.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Compute capability — see docs/results/env-report.md.
# Default 89 = RTX 4090 (Ada). Ampere = 80/86, Hopper = 90. Override with:
#   cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=<cc>
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

string(APPEND CMAKE_CUDA_FLAGS " -O3 --use_fast_math -lineinfo")

include_directories(${CMAKE_SOURCE_DIR}/kernels)

# Pure-CUDA kernel library. microbench.cu files carry their own main() and are
# excluded here (they become standalone executables below).
file(GLOB_RECURSE KERNEL_SRC kernels/*.cu)
list(FILTER KERNEL_SRC EXCLUDE REGEX "microbench")
add_library(llmik_kernels STATIC ${KERNEL_SRC})
set_target_properties(llmik_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Standalone microbenchmarks — uncomment each as the kernel + its microbench.cu
# land (Phase 1/2/3). Each is a pure-CUDA timing harness for one kernel.
#
# add_executable(bench_attention kernels/attention/microbench.cu)
# target_link_libraries(bench_attention PRIVATE llmik_kernels)
#
# add_executable(bench_kv_cache  kernels/kv_cache/microbench.cu)
# target_link_libraries(bench_kv_cache PRIVATE llmik_kernels)
#
# add_executable(bench_quant     kernels/quant/microbench.cu)
# target_link_libraries(bench_quant PRIVATE llmik_kernels)
