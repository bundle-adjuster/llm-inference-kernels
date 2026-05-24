// Torch glue layer: unpack torch::Tensor -> raw device pointers -> the
// pure-CUDA launchers in kernels/. Keep this layer thin; all kernel logic
// stays in the .cu files so they also build standalone via CMake.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention/fused_attention.cuh"
#include "kv_cache/kv_compress.cuh"


// Fused decode attention with fp16 KV (current Phase 1 v3 kernel).
// Inputs: q [b, n_heads, d], k/v [b, n_kv_heads, s, d], all fp16, contiguous.
// Returns: out [b, n_heads, d] fp16. softmax_scale is typically 1/sqrt(d).
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


// Quantize [batch, n_kv_heads, seqlen, head_dim] fp16 → int8 per-token, with
// fp16 scales of shape [batch, n_kv_heads, seqlen]. Returns (q, scale).
std::tuple<torch::Tensor, torch::Tensor>
quantize_per_token(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [batch, n_kv_heads, seqlen, head_dim]");

    const int batch       = x.size(0);
    const int n_kv_heads  = x.size(1);
    const int seqlen      = x.size(2);
    const int head_dim    = x.size(3);
    TORCH_CHECK(head_dim == 128, "only head_dim=128 supported");

    auto q     = torch::empty({batch, n_kv_heads, seqlen, head_dim},
                              x.options().dtype(torch::kInt8));
    auto scale = torch::empty({batch, n_kv_heads, seqlen},
                              x.options().dtype(torch::kHalf));

    launch_quantize_per_token(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        q.data_ptr<int8_t>(),
        reinterpret_cast<half*>(scale.data_ptr<at::Half>()),
        batch, n_kv_heads, seqlen, head_dim,
        at::cuda::getCurrentCUDAStream());

    return std::make_tuple(q, scale);
}


// Decode attention reading INT8-quantized KV with fused per-token dequant
// (Phase 2b). k_q/v_q are int8; k_scale/v_scale are fp16, shape [b, n_kv, s].
// Returns: out [b, n_heads, d] fp16.
torch::Tensor decode_attention_int8(torch::Tensor q,
                                    torch::Tensor k_q, torch::Tensor k_scale,
                                    torch::Tensor v_q, torch::Tensor v_scale,
                                    double softmax_scale) {
    TORCH_CHECK(q.is_cuda() && k_q.is_cuda() && k_scale.is_cuda()
                && v_q.is_cuda() && v_scale.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kHalf,
                "q must be fp16");
    TORCH_CHECK(k_q.scalar_type() == torch::kInt8
                && v_q.scalar_type() == torch::kInt8,
                "k_q and v_q must be int8");
    TORCH_CHECK(k_scale.scalar_type() == torch::kHalf
                && v_scale.scalar_type() == torch::kHalf,
                "k_scale and v_scale must be fp16");
    TORCH_CHECK(q.is_contiguous() && k_q.is_contiguous() && k_scale.is_contiguous()
                && v_q.is_contiguous() && v_scale.is_contiguous(),
                "all tensors must be contiguous");

    // q: [batch, n_heads, head_dim]
    // k_q/v_q: [batch, n_kv_heads, seqlen_kv, head_dim]
    // k_scale/v_scale: [batch, n_kv_heads, seqlen_kv]
    const int batch      = q.size(0);
    const int n_heads    = q.size(1);
    const int head_dim   = q.size(2);
    const int n_kv_heads = k_q.size(1);
    const int seqlen_kv  = k_q.size(2);

    auto out = torch::empty_like(q);

    launch_decode_attention_int8(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        k_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(k_scale.data_ptr<at::Half>()),
        v_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
        static_cast<float>(softmax_scale),
        at::cuda::getCurrentCUDAStream());

    return out;
}


// KIVI K-side quantization: INT4 per-channel, groupwise along seqlen.
// x: [b, n_kv, s, d] fp16. Returns (q [b, n_kv, s, d/2] int8 packed,
// scale [b, n_kv, ceil(s/group_size), d] fp16). group_size = 32 in KIVI.
std::tuple<torch::Tensor, torch::Tensor>
quantize_k_per_channel_groupwise_int4(torch::Tensor x, int64_t group_size) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [batch, n_kv_heads, seqlen, head_dim]");

    const int batch      = x.size(0);
    const int n_kv_heads = x.size(1);
    const int seqlen     = x.size(2);
    const int head_dim   = x.size(3);
    TORCH_CHECK(head_dim == 128, "only head_dim=128 supported");
    TORCH_CHECK(group_size > 0, "group_size must be positive");
    const int n_groups = (seqlen + group_size - 1) / group_size;

    auto q     = torch::empty({batch, n_kv_heads, seqlen, head_dim / 2},
                              x.options().dtype(torch::kInt8));
    auto scale = torch::empty({batch, n_kv_heads, n_groups, head_dim},
                              x.options().dtype(torch::kHalf));

    launch_quantize_k_per_channel_groupwise_int4(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        q.data_ptr<int8_t>(),
        reinterpret_cast<half*>(scale.data_ptr<at::Half>()),
        batch, n_kv_heads, seqlen, head_dim,
        static_cast<int>(group_size), n_groups,
        at::cuda::getCurrentCUDAStream());

    return std::make_tuple(q, scale);
}


// KIVI V-side quantization: INT4 per-token, packed.
// x: [b, n_kv, s, d] fp16. Returns (q [b, n_kv, s, d/2] int8 packed,
// scale [b, n_kv, s] fp16) — one fp16 scale per token.
std::tuple<torch::Tensor, torch::Tensor>
quantize_v_per_token_int4(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be [batch, n_kv_heads, seqlen, head_dim]");

    const int batch      = x.size(0);
    const int n_kv_heads = x.size(1);
    const int seqlen     = x.size(2);
    const int head_dim   = x.size(3);
    TORCH_CHECK(head_dim == 128, "only head_dim=128 supported");

    auto q     = torch::empty({batch, n_kv_heads, seqlen, head_dim / 2},
                              x.options().dtype(torch::kInt8));
    auto scale = torch::empty({batch, n_kv_heads, seqlen},
                              x.options().dtype(torch::kHalf));

    launch_quantize_v_per_token_int4(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        q.data_ptr<int8_t>(),
        reinterpret_cast<half*>(scale.data_ptr<at::Half>()),
        batch, n_kv_heads, seqlen, head_dim,
        at::cuda::getCurrentCUDAStream());

    return std::make_tuple(q, scale);
}


// Decode attention reading KIVI INT4 KV with fused per-channel-K +
// per-token-V dequant (Phase 2c). k_q/v_q are packed int4 (one int8
// byte = 2 nibbles); k_scale is [b, n_kv, n_groups, d] fp16; v_scale
// is [b, n_kv, s] fp16. group_size must match what the quantize
// kernel used (32 in KIVI). Returns: out [b, n_heads, d] fp16.
torch::Tensor decode_attention_int4(torch::Tensor q,
                                    torch::Tensor k_q, torch::Tensor k_scale,
                                    torch::Tensor v_q, torch::Tensor v_scale,
                                    int64_t group_size, double softmax_scale) {
    TORCH_CHECK(q.is_cuda() && k_q.is_cuda() && k_scale.is_cuda()
                && v_q.is_cuda() && v_scale.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kHalf,
                "q must be fp16");
    TORCH_CHECK(k_q.scalar_type() == torch::kInt8
                && v_q.scalar_type() == torch::kInt8,
                "k_q and v_q must be int8 (packed int4)");
    TORCH_CHECK(k_scale.scalar_type() == torch::kHalf
                && v_scale.scalar_type() == torch::kHalf,
                "k_scale and v_scale must be fp16");
    TORCH_CHECK(q.is_contiguous() && k_q.is_contiguous() && k_scale.is_contiguous()
                && v_q.is_contiguous() && v_scale.is_contiguous(),
                "all tensors must be contiguous");

    // q: [batch, n_heads, head_dim]
    // k_q/v_q: [batch, n_kv_heads, seqlen_kv, head_dim/2] (packed)
    // k_scale: [batch, n_kv_heads, n_groups, head_dim]
    // v_scale: [batch, n_kv_heads, seqlen_kv]
    const int batch      = q.size(0);
    const int n_heads    = q.size(1);
    const int head_dim   = q.size(2);
    const int n_kv_heads = k_q.size(1);
    const int seqlen_kv  = k_q.size(2);
    const int n_groups   = k_scale.size(2);

    auto out = torch::empty_like(q);

    launch_decode_attention_int4(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        k_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(k_scale.data_ptr<at::Half>()),
        v_q.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(v_scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
        static_cast<int>(group_size), n_groups,
        static_cast<float>(softmax_scale),
        at::cuda::getCurrentCUDAStream());

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attention", &decode_attention,
          "Fused decode attention (fp16 KV, custom CUDA kernel)");
    m.def("quantize_per_token", &quantize_per_token,
          "INT8 per-token symmetric quantization for KV-cache");
    m.def("decode_attention_int8", &decode_attention_int8,
          "Fused decode attention reading INT8 KV with per-token dequant");
    m.def("quantize_k_per_channel_groupwise_int4",
          &quantize_k_per_channel_groupwise_int4,
          "INT4 per-channel groupwise quantization (KIVI K path)");
    m.def("quantize_v_per_token_int4",
          &quantize_v_per_token_int4,
          "INT4 per-token quantization, packed (KIVI V path)");
    m.def("decode_attention_int4", &decode_attention_int4,
          "Fused decode attention reading KIVI INT4 KV with fused dequant");
}
