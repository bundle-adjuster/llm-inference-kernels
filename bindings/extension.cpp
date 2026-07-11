// Torch glue layer: unpack torch::Tensor -> raw device pointers -> the
// pure-CUDA launchers in kernels/. Keep this layer thin; all kernel logic
// stays in the .cu files so they also build standalone via CMake.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention/fused_attention.cuh"
#include "kv_cache/kv_compress.cuh"
#include "quant/quant_matmul.cuh"
#include "fused/fused_ops.cuh"


// Fused decode attention with fp16 KV.
// Inputs: q [b, n_heads, d], k/v [b, n_kv_heads, s, d], all fp16, contiguous.
// Returns: out [b, n_heads, d] fp16. softmax_scale is typically 1/sqrt(d).
//
// Dispatches to the v6 FlashDecoding split-K kernel (fixes v3's occupancy wall,
// Phase 7) whenever splitting helps; falls back to the v3 single-block kernel
// when the sequence is too short to split usefully. The old v3 kernel is still
// reachable directly as `decode_attention_v3` for before/after benchmarking.
// `seqlen` is the live KV length. Default (-1) treats k/v as a contiguous
// [batch, n_kv_heads, S, head_dim] cache of length S = k.size(2). Passing a live
// length < k.size(2) reads the [0, seqlen) prefix of a *preallocated* cache in
// place (stride = k.size(2)) — this is how the decode path skips the per-step KV
// `torch.cat`, which SDPA could not do (it fell back to the math backend).
torch::Tensor decode_attention(torch::Tensor q, torch::Tensor k,
                               torch::Tensor v, double softmax_scale,
                               int64_t seqlen = -1) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "fp16 only for now");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "inputs must be contiguous");

    // q: [batch, n_heads, head_dim]
    // k/v: [batch, n_kv_heads, kv_buf_len, head_dim]
    const int batch      = q.size(0);
    const int n_heads    = q.size(1);
    const int head_dim   = q.size(2);
    const int n_kv_heads = k.size(1);
    const int kv_buf_len = k.size(2);                 // allocated stride
    const int seqlen_kv  = (seqlen > 0) ? static_cast<int>(seqlen) : kv_buf_len;
    TORCH_CHECK(seqlen_kv <= kv_buf_len, "seqlen exceeds allocated cache length");
    const bool strided   = (seqlen_kv != kv_buf_len);

    auto out = torch::empty_like(q);
    auto stream = at::cuda::getCurrentCUDAStream();

    int n_splits = decode_attention_n_splits(batch, n_heads, seqlen_kv);

    // v3 assumes a contiguous cache (stride == length); only split-K carries the
    // separate kv_buf_len. So take the v3 fast path only for a contiguous cache.
    if (n_splits <= 1 && !strided) {
        launch_decode_attention(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
            static_cast<float>(softmax_scale), stream);
        return out;
    }
    if (n_splits < 1) n_splits = 1;

    // Split-K scratch (fp32). torch's caching allocator makes the per-call
    // allocation effectively free after warmup.
    auto f32 = q.options().dtype(torch::kFloat32);
    auto partial_o = torch::empty({batch, n_heads, n_splits, head_dim}, f32);
    auto partial_m = torch::empty({batch, n_heads, n_splits}, f32);
    auto partial_l = torch::empty({batch, n_heads, n_splits}, f32);

    launch_decode_attention_splitk(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        partial_o.data_ptr<float>(), partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        batch, n_heads, n_kv_heads, seqlen_kv, head_dim,
        kv_buf_len, n_splits, static_cast<float>(softmax_scale), stream);

    return out;
}


// The Phase 1 v3 kernel, exposed directly. Single-warp block per (batch, head)
// streaming the whole KV sequence — kept for reproducing the pre-Phase-7
// baseline and measuring the v3 -> v6 split-K speedup. Prefer decode_attention.
torch::Tensor decode_attention_v3(torch::Tensor q, torch::Tensor k,
                                  torch::Tensor v, double softmax_scale) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "q, k, v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kHalf, "fp16 only for now");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "inputs must be contiguous");

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


// Phase 3 W4A16 GEMM. act: [M, K] fp16. weight_packed: [K/8, N] int32
// (the bit pattern is uint32; torch has no native uint32). scale:
// [n_groups, N] fp16 with n_groups = K / group_size. group_size must
// be a multiple of 8 (the int4 pack width along K).
// Returns: out [M, N] fp16.
torch::Tensor w4a16_gemm(torch::Tensor act,
                         torch::Tensor weight_packed,
                         torch::Tensor scale,
                         int64_t group_size) {
    TORCH_CHECK(act.is_cuda() && weight_packed.is_cuda() && scale.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(act.scalar_type() == torch::kHalf, "act must be fp16");
    TORCH_CHECK(weight_packed.scalar_type() == torch::kInt32,
                "weight_packed must be int32 (uint32 bit pattern)");
    TORCH_CHECK(scale.scalar_type() == torch::kHalf, "scale must be fp16");
    TORCH_CHECK(act.is_contiguous() && weight_packed.is_contiguous()
                && scale.is_contiguous(), "all tensors must be contiguous");
    TORCH_CHECK(act.dim() == 2 && weight_packed.dim() == 2 && scale.dim() == 2,
                "all inputs must be 2D");

    const int M       = act.size(0);
    const int K       = act.size(1);
    const int K_packs = weight_packed.size(0);
    const int N       = weight_packed.size(1);
    const int n_groups = scale.size(0);
    TORCH_CHECK(K == K_packs * 8,
                "act.K must equal weight_packed.K_packs * 8");
    TORCH_CHECK(scale.size(1) == N, "scale and weight_packed N mismatch");
    TORCH_CHECK(group_size > 0 && group_size % 8 == 0,
                "group_size must be a positive multiple of 8");
    TORCH_CHECK(K == n_groups * static_cast<int>(group_size),
                "K must equal n_groups * group_size");

    auto out = torch::empty({M, N}, act.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int n_splits = w4a16_n_splits(M, N, K, static_cast<int>(group_size));
    if (n_splits > 1) {
        // Batched decode (2 <= M <= 16): tensor-core + split-K over K (Phase 9),
        // with an fp32 accumulator the split partials atomicAdd into.
        auto acc = torch::empty({M, N}, act.options().dtype(torch::kFloat32));
        launch_w4a16_gemm_splitk(
            reinterpret_cast<const half*>(act.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(weight_packed.data_ptr<int32_t>()),
            reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
            acc.data_ptr<float>(),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            M, N, K, static_cast<int>(group_size), n_splits, stream);
        return out;
    }

    launch_w4a16_gemm(
        reinterpret_cast<const half*>(act.data_ptr<at::Half>()),
        reinterpret_cast<const uint32_t*>(weight_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, N, K, static_cast<int>(group_size), stream);

    return out;
}


// Materialize the full fp16 weight [K, N] from packed int4 + groupwise scales,
// in one CUDA pass. For the prefill path (large M), where a dequant + cuBLAS
// GEMM beats the decode-shaped quantized kernel. weight_packed: [K/8, N] int32.
torch::Tensor w4a16_dequantize(torch::Tensor weight_packed, torch::Tensor scale,
                               int64_t group_size) {
    TORCH_CHECK(weight_packed.is_cuda() && scale.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(weight_packed.scalar_type() == torch::kInt32, "weight_packed int32");
    TORCH_CHECK(scale.scalar_type() == torch::kHalf, "scale fp16");
    TORCH_CHECK(weight_packed.is_contiguous() && scale.is_contiguous(), "contiguous");

    const int K = weight_packed.size(0) * 8;
    const int N = weight_packed.size(1);
    TORCH_CHECK(scale.size(1) == N, "scale/weight N mismatch");
    TORCH_CHECK(group_size > 0 && K % static_cast<int>(group_size) == 0,
                "K must be a multiple of group_size");

    auto out = torch::empty({K, N}, scale.options());
    launch_w4a16_dequantize(
        reinterpret_cast<const uint32_t*>(weight_packed.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(scale.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        K, N, static_cast<int>(group_size),
        at::cuda::getCurrentCUDAStream());
    return out;
}


// Fused RMSNorm (Phase 10). x: [..., H] fp16; weight: [H] fp16. Returns same
// shape as x. eps matches the model's rms_norm_eps.
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf && weight.scalar_type() == torch::kHalf,
                "fp16 only");
    const int H = x.size(-1);
    TORCH_CHECK(weight.numel() == H, "weight length must match x last dim");
    auto xc = x.contiguous();
    auto x2 = xc.reshape({-1, H});
    const int M = x2.size(0);
    auto out = torch::empty_like(x2);
    launch_rmsnorm(
        reinterpret_cast<const half*>(x2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, H, static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
    return out.reshape(x.sizes());
}

// Fused SwiGLU (Phase 10): silu(gate) * up, elementwise. Same shape in/out.
torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(gate.scalar_type() == torch::kHalf && up.scalar_type() == torch::kHalf,
                "fp16 only");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
    auto g = gate.contiguous();
    auto u = up.contiguous();
    auto out = torch::empty_like(g);
    launch_silu_mul(
        reinterpret_cast<const half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(u.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(g.numel()), at::cuda::getCurrentCUDAStream());
    return out;
}


// Fused RoPE (Phase 10). x: [B, H, S, D] fp16; cos/sin: [B, S, D] fp16.
// Returns x rotated: x*cos + rotate_half(x)*sin.
torch::Tensor rope(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    TORCH_CHECK(x.is_cuda() && cos.is_cuda() && sin.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "fp16 only");
    TORCH_CHECK(x.dim() == 4, "x must be [B, H, S, D]");
    const int B = x.size(0), H = x.size(1), S = x.size(2), D = x.size(3);
    TORCH_CHECK(D % 2 == 0, "head_dim must be even");
    auto xc = x.contiguous();
    // cos/sin arrive as [B, S, D] or broadcastable [S, D] / [1, S, D]. Normalize
    // to a contiguous [B, S, D] (the broadcast copy is tiny).
    auto norm_cs = [&](torch::Tensor t) {
        t = t.contiguous();
        const long need = static_cast<long>(B) * S * D;
        if (t.numel() == need) return t.reshape({B, S, D});
        TORCH_CHECK(t.numel() == static_cast<long>(S) * D,
                    "cos/sin must be [B,S,D] or broadcastable [S,D]/[1,S,D]");
        return t.reshape({1, S, D}).expand({B, S, D}).contiguous();
    };
    auto cc = norm_cs(cos);
    auto sc = norm_cs(sin);
    auto out = torch::empty_like(xc);
    launch_rope(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        B, H, S, D, at::cuda::getCurrentCUDAStream());
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_attention", &decode_attention,
          "Fused decode attention (fp16 KV; v6 FlashDecoding split-K, v3 fallback)",
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("softmax_scale"), pybind11::arg("seqlen") = -1);
    m.def("decode_attention_v3", &decode_attention_v3,
          "Phase 1 v3 decode attention (single-warp block); pre-Phase-7 baseline");
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
    m.def("w4a16_gemm", &w4a16_gemm,
          "W4A16 quantized matmul: fp16 act @ INT4 packed weight (decode-shape kernel)");
    m.def("w4a16_dequantize", &w4a16_dequantize,
          "Materialize fp16 weight [K,N] from packed int4 + groupwise scales (prefill path)");
    m.def("rmsnorm", &rmsnorm, "Fused RMSNorm over the last dim (fp16)");
    m.def("silu_mul", &silu_mul, "Fused SwiGLU: silu(gate) * up (fp16)");
    m.def("rope", &rope, "Fused RoPE: x*cos + rotate_half(x)*sin (fp16)");
}
