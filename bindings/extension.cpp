// Torch glue layer: unpack torch::Tensor -> raw device pointers -> the
// pure-CUDA launchers in kernels/. Keep this layer thin; all kernel logic
// stays in the .cu files so they also build standalone via CMake.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention/fused_attention.cuh"
// kv_cache and quant ops are registered here in Phase 2 / Phase 3.

torch::Tensor decode_attention(torch::Tensor q, torch::Tensor k,
                               torch::Tensor v, double softmax_scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "fp16 only for now");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "inputs must be contiguous");

    // q: [batch, n_heads, head_dim]
    // k/v: [batch, n_kv_heads, seqlen_kv, head_dim]
    const int batch      = q.size(0);
    const int n_heads    = q.size(1);
    const int head_dim   = q.size(2);
    const int n_kv_heads = k.size(1);
    const int seqlen_kv  = k.size(2);

    auto out = torch::empty_like(q);

    launch_decode_attention(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
        static_cast<float>(softmax_scale),
        at::cuda::getCurrentCUDAStream());

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attention", &decode_attention,
          "Fused decode attention (custom CUDA kernel)");
}
