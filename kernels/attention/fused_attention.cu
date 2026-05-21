#include "attention/fused_attention.cuh"
#include "common/cuda_utils.cuh"

// TODO(Phase 1) — implement the decode attention kernel.
// Roadmap (one commit + one RESULTS.md row per step, see docs/01):
//   v0  naive: one block per (batch, head), two-pass softmax  <-- start here
//   v1  online (streaming) softmax, single pass
//   v2  warp-level reductions for dot-products and m/l
//   v3  vectorized 128-bit coalesced KV loads
//   v4  split-K over the KV sequence + combine (FlashDecoding)
//   v5  cp.async double-buffering of KV tiles

void launch_decode_attention(
    const half* q, const half* k, const half* v, half* out,
    int batch, int n_heads, int n_kv_heads, int seqlen_kv, int head_dim,
    float softmax_scale, cudaStream_t stream) {
    // TODO(Phase 1b): launch the v0 kernel here.
    (void)q; (void)k; (void)v; (void)out;
    (void)batch; (void)n_heads; (void)n_kv_heads; (void)seqlen_kv;
    (void)head_dim; (void)softmax_scale; (void)stream;
}
