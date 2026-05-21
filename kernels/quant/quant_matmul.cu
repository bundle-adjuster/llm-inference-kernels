#include "quant/quant_matmul.cuh"
#include "common/cuda_utils.cuh"

// TODO(Phase 3) — W4A16 GEMM. See docs/03-quantized-matmul.md:
//   v0  naive: unpack INT4, dequant in registers, accumulate  <-- start here
//   v1  group-wise scales, shared-memory staging
//   v2  vectorized INT4 unpack, coalesced weight loads
//   v3  Tensor Core MMA path for compute-bound (prefill) shapes
//
// Primary target is the decode shape (M small) — memory-bound on weight
// traffic, where 4-bit weights cut HBM reads ~4x.

void launch_w4a16_gemm(
    const half* act, const uint32_t* weight_packed, const half* scales,
    half* out, int M, int N, int K, int group_size, cudaStream_t stream) {
    // TODO(Phase 3).
    (void)act; (void)weight_packed; (void)scales; (void)out;
    (void)M; (void)N; (void)K; (void)group_size; (void)stream;
}
