#include "kv_cache/kv_compress.cuh"
#include "common/cuda_utils.cuh"

// TODO(Phase 2) — KV-cache quantization. See docs/02-kv-cache-compression.md:
//   - INT8 quantize kernel (group-wise scales)
//   - INT4 path: per-channel K scales, per-token V scales (KIVI-style)
//   - the fused dequant lives in kernels/attention (Track 1 kernel reads
//     packed K/V directly).

void launch_kv_quantize(
    const half* kv_fp16, int8_t* kv_quant, half* scales,
    int n_kv_heads, int head_dim, int bits, cudaStream_t stream) {
    // TODO(Phase 2).
    (void)kv_fp16; (void)kv_quant; (void)scales;
    (void)n_kv_heads; (void)head_dim; (void)bits; (void)stream;
}
