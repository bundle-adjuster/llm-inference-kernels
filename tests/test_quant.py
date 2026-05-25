"""Correctness tests for the W4A16 quantized matmul reference.

Pure PyTorch — no CUDA extension needed. These define the contract that
the Phase 3 CUDA kernels (kernels/quant/*.cu) must match.

Two properties to check:

1. **Round-trip**: `dequant(quant(W)) ≈ W` within `1/qmax` of the
   per-(group, output-channel) max. INT4 → qmax=7 → per-element error
   ≤ scale/2 ≈ absmax_per_group_per_channel / 14.

2. **Matmul quantization noise**: `act @ dequant(quant(W))` is close to
   `act @ W`. Per-element noise is bounded; over a length-K sum it
   averages roughly `√K · noise_floor`. We allow a slack consistent
   with that.
"""
from __future__ import annotations

import math

import pytest
import torch

from reference.quant_matmul_ref import (
    _QMAX_INT4,
    dequantize_weights_int4_groupwise,
    quantize_weights_int4_groupwise,
    quantized_matmul_ref,
    w4a16_matmul_ref,
)


# Llama 3 8B linear-layer shapes. These are the real (K, N) the CUDA
# kernel will target. K is the reduction axis, N is the output dim.
LLAMA_LAYER_SHAPES = [
    ("attn-qkv-or-out", 4096, 4096),    # K=4096, N=4096
    ("mlp-up-or-gate",  4096, 14336),   # K=4096, N=14336
    ("mlp-down",       14336,  4096),   # K=14336, N=4096
]


def _make_weight(K: int, N: int, *, seed: int = 0) -> torch.Tensor:
    """Random fp16 weight with Llama-like dynamic range.

    Real Llama weights are roughly std≈0.02 with occasional outliers.
    We use a wider std (0.1) and inject channel-correlated outliers so
    the groupwise structure has a chance to matter — without that, all
    groups end up with similar scales and the test stops being a real
    workout for the algorithm.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(K, N, generator=g, dtype=torch.float32) * 0.1
    # Inject a per-channel outlier in some groups (every ~9th column,
    # every ~7th K-position) so absmax per (group, channel) varies.
    W[::7, ::9] *= 5.0
    return W.to(torch.float16)


# ---- round-trip ----

@pytest.mark.parametrize("name,K,N", LLAMA_LAYER_SHAPES)
@pytest.mark.parametrize("group_size", [64, 128])
def test_quantize_dequantize_roundtrip(
    name: str, K: int, N: int, group_size: int,
) -> None:
    W = _make_weight(K, N)
    q, scale = quantize_weights_int4_groupwise(W, group_size=group_size)
    W_hat = dequantize_weights_int4_groupwise(q, scale, group_size).to(torch.float16)

    # Shape and dtype invariants.
    assert q.dtype == torch.int8
    assert q.shape == (K, N)
    assert scale.shape == ((K + group_size - 1) // group_size, N)
    assert q.min().item() >= -_QMAX_INT4 and q.max().item() <= _QMAX_INT4

    # Per-(group, output-channel) tolerance: error <= scale, with a
    # tiny slack for fp16 rounding on the way back. Same logic as the
    # Phase 2 per-channel groupwise test.
    n_groups = (K + group_size - 1) // group_size
    pad = n_groups * group_size - K
    pad_zeros = torch.zeros(pad, N) if pad else None
    W_padded = torch.cat([W, pad_zeros], dim=0) if pad else W
    per_gc_max = W_padded.view(n_groups, group_size, N).abs().amax(dim=1)  # [g, N]
    tol = per_gc_max / _QMAX_INT4

    err = (W_hat - W).abs()
    err_padded = torch.cat([err, pad_zeros], dim=0) if pad else err
    err_per_gc_max = err_padded.view(n_groups, group_size, N).amax(dim=1)
    assert (err_per_gc_max <= tol * 1.001).all(), (
        f"int4 K-groupwise W round-trip: max rel err "
        f"{(err_per_gc_max / tol.clamp(min=1e-8)).max().item():.3f}"
    )


# ---- matmul correctness ----

@pytest.mark.parametrize("name,K,N", LLAMA_LAYER_SHAPES)
@pytest.mark.parametrize("M", [1, 8, 32])
def test_quantized_matmul_close_to_fp16(
    name: str, K: int, N: int, M: int,
) -> None:
    """`act @ dequant(quant(W))` should be close to `act @ W`.

    Per-element noise floor on W is ~`scale/2 = absmax_per_group / 14`.
    The matmul sums K such noisy terms times fp16 activations of std≈1;
    the expected std of the noise on `out[m, n]` is roughly
    `sqrt(K) · (noise_floor) · std(act)`. We use a tolerance derived
    from that bound, generously inflated to absorb fp16 accumulation.
    """
    torch.manual_seed(0)
    W = _make_weight(K, N)
    act = torch.randn(M, K, dtype=torch.float16) * 1.0

    # Reference outputs: fp16 baseline + quantized.
    out_fp16 = (act.float() @ W.float()).to(torch.float16)
    out_q, q, scale = w4a16_matmul_ref(act, W, group_size=128)

    # Sanity: storage shape.
    n_groups = (K + 127) // 128
    assert q.shape == (K, N)
    assert scale.shape == (n_groups, N)

    # Tolerance: bound the matmul noise. The per-element noise on W is
    # ≤ scale/2 ≈ absmax_per_group / 14. Sum K such terms (with random
    # ±1 act values, modelled as iid) gives a std proportional to
    # sqrt(K) · noise_floor · std(act). We use a generous bound that
    # passes empirically on these shapes (max ~5-10% relative).
    abs_err = (out_q.float() - out_fp16.float()).abs()
    out_max = out_fp16.float().abs().amax().clamp(min=1e-3)
    rel_err_max = (abs_err / out_max).max().item()
    rel_err_mean = (abs_err / out_max).mean().item()

    # Quantization noise on each output element is a sum of K terms,
    # each bounded by `scale_group/2`. Over N output elements, the
    # *max* hits the tail (~5σ) at ~30% on these shapes; the *mean*
    # tracks the expected σ at a few %. These bounds are the INT4 noise
    # floor on synthetic gaussian weights with injected outliers — not
    # a CUDA-kernel correctness gate; that comes in Phase 3b/3c via the
    # dequant-equivalence test (CUDA output bit-equal to the reference
    # on the same dequantized weights).
    assert rel_err_max < 0.35, (
        f"{name} M={M}: max rel err {rel_err_max:.3f} > 0.35"
    )
    assert rel_err_mean < 0.06, (
        f"{name} M={M}: mean rel err {rel_err_mean:.3f} > 0.06"
    )


# ---- storage sanity ----

def test_int4_storage_quarter_of_fp16_when_packed() -> None:
    """W4A16 weight storage at 4 bits per element (packed) plus scales.

    Reference uses int8 containers (unpacked); the CUDA kernel will pack
    2 nibbles/byte. We compute the *theoretical packed* size and verify
    it lands ~1/4 of fp16 plus a small per-group scale overhead.
    """
    K, N = 4096, 14336
    group_size = 128
    W = _make_weight(K, N)
    q, scale = quantize_weights_int4_groupwise(W, group_size=group_size)

    # Theoretical packed-int4 weight bytes (4 bits per element).
    packed_w_bytes = (q.numel() + 1) // 2

    # Scale overhead (fp16).
    scales_bytes = scale.numel() * 2

    fp16_bytes = W.numel() * 2
    ratio = (packed_w_bytes + scales_bytes) / fp16_bytes

    # Expected: ~0.25 from weights + tiny overhead from scales
    # (one fp16 per group_size=128 K positions per output column,
    # so scales overhead = 2 / (group_size * 2) = 1/group_size of fp16).
    # Total: 0.25 + 1/128 = ~0.258.
    assert 0.24 <= ratio <= 0.27, (
        f"int4 W storage ratio {ratio:.4f} out of expected band"
    )


def test_quantize_handles_small_K_smoke() -> None:
    """A tiny shape just to exercise the partial-final-group path."""
    K, N = 100, 32   # K not a multiple of 128
    group_size = 128
    torch.manual_seed(0)
    W = torch.randn(K, N, dtype=torch.float16) * 0.5
    q, scale = quantize_weights_int4_groupwise(W, group_size=group_size)
    W_hat = dequantize_weights_int4_groupwise(q, scale, group_size).to(torch.float16)

    n_groups = (K + group_size - 1) // group_size
    assert q.shape == (K, N)
    assert scale.shape == (n_groups, N)

    # Round-trip stays bounded.
    err = (W_hat - W).abs()
    per_token_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    assert (err / per_token_max).max().item() < 0.3
