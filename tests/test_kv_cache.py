"""Correctness tests for the KV-cache quantization reference.

Pure PyTorch — no CUDA extension needed. These define the contract that
the Phase 2 CUDA kernels (kernels/kv_cache/*.cu) must match.

Round-trip property: for any input `x`, dequant(quant(x)) reconstructs `x`
within a tolerance bounded by the quantization step size.

For symmetric integer quantization with `qmax = 2^(bits-1) - 1`:
  per-element error  ≤  scale / 2  =  (absmax_of_axis / qmax) / 2

We allow 1.0 / qmax of the per-axis max as a generous slack (covers
rounding direction + the clamp guard); INT8 lands at <1%, INT4 at <15%.
"""
from __future__ import annotations

import math

import pytest
import torch

from reference.attention_ref import decode_attention
from reference.kv_cache_ref import (
    dequantize_kv_int8_per_token,
    dequantize_kv_kivi_int4,
    dequantize_per_channel_groupwise,
    dequantize_per_token,
    quantize_kv_int8_per_token,
    quantize_kv_kivi_int4,
    quantize_per_channel_groupwise,
    quantize_per_token,
)


# Llama 3 8B head config — same shapes the Phase 1 attention kernel uses.
N_HEADS_KV = 8
HEAD_DIM = 128


def _make_kv(batch: int, seqlen: int, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random K, V with a realistic dynamic range.

    Std-normal scaled to roughly the magnitudes we see in Llama activations.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randn(batch, N_HEADS_KV, seqlen, HEAD_DIM, generator=g, dtype=torch.float32)
    v = torch.randn(batch, N_HEADS_KV, seqlen, HEAD_DIM, generator=g, dtype=torch.float32)
    # Inject a few per-channel outliers into K — this is the pattern that
    # motivates per-channel scales (KIVI). Without it, per-token and
    # per-channel modes look essentially the same.
    k[..., ::17] *= 3.5
    return k, v


# ---- per-axis round-trip ----

@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024)])
def test_per_token_roundtrip(bits: int, batch: int, seqlen: int) -> None:
    qmax = (1 << (bits - 1)) - 1
    x, _ = _make_kv(batch, seqlen)
    q, scale = quantize_per_token(x, bits)
    x_recon = dequantize_per_token(q, scale)

    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert scale.shape == x.shape[:-1]

    # Per-token absolute tolerance: max element error ≤ scale = absmax / qmax.
    per_token_max = x.abs().amax(dim=-1, keepdim=True)        # [..., 1]
    tol = per_token_max / qmax
    abs_err = (x_recon - x).abs()
    assert (abs_err <= tol * 1.001).all(), (
        f"per-token bits={bits}: max relative error "
        f"{(abs_err / tol.clamp(min=1e-8)).max().item():.3f} > 1.0 (qmax={qmax})"
    )

    # All integer values stay in symmetric range.
    assert q.min().item() >= -qmax and q.max().item() <= qmax


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024), (2, 100)])  # 100 to exercise pad
def test_per_channel_groupwise_roundtrip(
    bits: int, group_size: int, batch: int, seqlen: int
) -> None:
    qmax = (1 << (bits - 1)) - 1
    x, _ = _make_kv(batch, seqlen)
    q, scale = quantize_per_channel_groupwise(x, bits, group_size)
    x_recon = dequantize_per_channel_groupwise(q, scale, group_size)

    n_groups = (seqlen + group_size - 1) // group_size
    assert q.shape == x.shape
    assert scale.shape == (batch, N_HEADS_KV, n_groups, HEAD_DIM)

    # Per-(group, channel) tolerance. Both the input and the error tensor
    # must be padded to a multiple of group_size before reshaping into
    # (n_groups, group_size) — the last group may be short.
    pad = n_groups * group_size - seqlen
    pad_zeros = torch.zeros(batch, N_HEADS_KV, pad, HEAD_DIM) if pad else None
    x_padded = torch.cat([x, pad_zeros], dim=2) if pad else x
    per_gc_max = x_padded.view(
        batch, N_HEADS_KV, n_groups, group_size, HEAD_DIM
    ).abs().amax(dim=3)                                       # [b, kv, ng, d]
    tol = per_gc_max / qmax

    abs_err = (x_recon - x).abs()
    err_padded = torch.cat([abs_err, pad_zeros], dim=2) if pad else abs_err
    err_per_gc_max = err_padded.view(
        batch, N_HEADS_KV, n_groups, group_size, HEAD_DIM
    ).amax(dim=3)                                             # [b, kv, ng, d]
    assert (err_per_gc_max <= tol * 1.001).all(), (
        f"per-channel-groupwise bits={bits} g={group_size}: "
        f"max relative error {(err_per_gc_max / tol.clamp(min=1e-8)).max().item():.3f}"
    )


# ---- KIVI presets ----

@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024)])
def test_int8_per_token_preset_roundtrip(batch: int, seqlen: int) -> None:
    k, v = _make_kv(batch, seqlen)
    k_q, k_s, v_q, v_s = quantize_kv_int8_per_token(k, v)
    k_rec, v_rec = dequantize_kv_int8_per_token(k_q, k_s, v_q, v_s, out_dtype=torch.float32)

    # INT8 per-token: errors well under 1% of per-token max.
    k_max = k.abs().amax(dim=-1, keepdim=True)
    v_max = v.abs().amax(dim=-1, keepdim=True)
    assert ((k_rec - k).abs() <= k_max / 127 * 1.001).all()
    assert ((v_rec - v).abs() <= v_max / 127 * 1.001).all()


@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024)])
def test_kivi_int4_preset_roundtrip(batch: int, seqlen: int) -> None:
    group_size = 32
    k, v = _make_kv(batch, seqlen)
    k_q, k_s, v_q, v_s = quantize_kv_kivi_int4(k, v, group_size=group_size)
    k_rec, v_rec = dequantize_kv_kivi_int4(
        k_q, k_s, v_q, v_s, group_size=group_size, out_dtype=torch.float32
    )

    # INT4 has qmax=7, so per-axis tolerance is roughly max/7.
    n_groups = (seqlen + group_size - 1) // group_size
    pad = n_groups * group_size - seqlen
    k_padded = (
        torch.cat([k, torch.zeros(batch, N_HEADS_KV, pad, HEAD_DIM)], dim=2)
        if pad else k
    )
    k_per_gc_max = k_padded.view(
        batch, N_HEADS_KV, n_groups, group_size, HEAD_DIM
    ).abs().amax(dim=3)                                       # [b, kv, ng, d]
    k_err_per_gc = (k_rec - k).abs().view(
        batch, N_HEADS_KV, n_groups, -1, HEAD_DIM
    ).amax(dim=3)
    assert (k_err_per_gc <= k_per_gc_max / 7 * 1.001).all()

    v_max = v.abs().amax(dim=-1, keepdim=True)
    assert ((v_rec - v).abs() <= v_max / 7 * 1.001).all()


# ---- shape / storage sanity ----

def test_int8_storage_is_smaller_than_fp16() -> None:
    """Half the bytes per K/V element; scales add a small overhead."""
    batch, seqlen = 4, 4096
    k, v = _make_kv(batch, seqlen)
    fp16_bytes = (k.element_size() // 2 + v.element_size() // 2) * k.numel()  # fp16 = 2 B

    k_q, k_s, v_q, v_s = quantize_kv_int8_per_token(k, v)
    int8_bytes = (
        k_q.numel() * k_q.element_size()
        + v_q.numel() * v_q.element_size()
        + k_s.numel() * 2   # fp16 scales
        + v_s.numel() * 2
    )

    # int8 = 1 B/elem; scales are head_dim=128 less frequent than KV elements,
    # so overhead is ~1/128 → total roughly 0.51× the fp16 size.
    ratio = int8_bytes / fp16_bytes
    assert 0.50 <= ratio <= 0.52, f"int8 storage ratio {ratio:.4f} out of expected band"


def test_int4_storage_quarter_when_packed() -> None:
    """Reference stores 4-bit values in int8 containers (no packing yet);
    a packed CUDA path would land at ~1/4 of fp16.

    Here we just check the *scales overhead* on top of theoretical 4-bit
    K/V is small. Real packing happens in the CUDA kernel; the reference
    uses int8 storage for clarity.
    """
    batch, seqlen, group_size = 4, 4096, 32
    k, v = _make_kv(batch, seqlen)
    k_q, k_s, v_q, v_s = quantize_kv_kivi_int4(k, v, group_size=group_size)

    # Theoretical packed-int4 K/V bytes (4 bits/elem = 0.5 B):
    packed_kv_bytes = (k_q.numel() + v_q.numel()) // 2

    # Scales overhead (fp16):
    scales_bytes = (k_s.numel() + v_s.numel()) * 2

    fp16_bytes = (k.numel() + v.numel()) * 2
    ratio = (packed_kv_bytes + scales_bytes) / fp16_bytes

    # With group_size=32 per-channel K + per-token V, scale overhead is
    # K_scales: 1 scale per (n_groups, head_dim) = head_dim / group_size of K bits;
    # V_scales: 1 scale per token = 1 / head_dim of V bits.
    # Total target band: ~0.26–0.30× fp16.
    assert 0.25 <= ratio <= 0.32, f"int4 storage ratio {ratio:.4f} out of band"


# ---- CUDA kernels vs PyTorch reference ----

llmik = pytest.importorskip(
    "llmik_cuda",
    reason="build the extension: python setup.py build_ext --inplace",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required",
)


@requires_cuda
@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024), (8, 4096)])
def test_cuda_quantize_per_token_matches_reference(batch: int, seqlen: int) -> None:
    """The CUDA kernel must match the PyTorch reference exactly (modulo
    a few rounding-direction ties — see tolerance below)."""
    torch.manual_seed(0)
    x = torch.randn(batch, N_HEADS_KV, seqlen, HEAD_DIM,
                    device="cuda", dtype=torch.float16)

    q_cuda, s_cuda = llmik.quantize_per_token(x)
    q_ref, s_ref   = quantize_per_token(x.float(), bits=8)
    q_ref = q_ref.to("cuda")
    s_ref = s_ref.to("cuda").to(torch.float16)

    # Scales: same fp32 computation → should be near-bitwise identical at fp16.
    torch.testing.assert_close(s_cuda.float(), s_ref.float(), rtol=1e-3, atol=1e-3)

    # Quantized values: round-to-nearest-even both sides, so they agree at
    # all non-tie positions. Allow off-by-one at ties.
    diff = (q_cuda.int() - q_ref.int()).abs()
    assert (diff <= 1).all(), (
        f"CUDA vs reference int8 diff > 1 at {(diff > 1).sum().item()} positions"
    )
    assert diff.float().mean().item() < 0.05, (
        f"avg |diff| {diff.float().mean().item():.4f} too high — "
        "more than 5% ties is suspicious"
    )


@requires_cuda
@pytest.mark.parametrize("batch,seqlen_kv", [(1, 128), (4, 1024), (8, 2048)])
def test_cuda_decode_attention_int8_matches_dequant_reference(
    batch: int, seqlen_kv: int
) -> None:
    """End-to-end: quantize fp16 K/V via the CUDA kernel, run the INT8
    attention kernel, compare against the PyTorch reference attention
    on the *dequantized* K/V. The two paths see identical fp16 K/V (modulo
    quantization noise), so they should agree within fp16 round-off."""
    torch.manual_seed(0)
    n_heads, n_kv_heads, head_dim = 32, N_HEADS_KV, HEAD_DIM
    scale = 1.0 / math.sqrt(head_dim)

    q   = torch.randn(batch, n_heads,    head_dim,
                      device="cuda", dtype=torch.float16)
    k   = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                      device="cuda", dtype=torch.float16)
    v   = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                      device="cuda", dtype=torch.float16)

    # CUDA path: quantize K, V via the CUDA kernel; run INT8 attention.
    k_q, k_s = llmik.quantize_per_token(k)
    v_q, v_s = llmik.quantize_per_token(v)
    out_cuda = llmik.decode_attention_int8(q, k_q, k_s, v_q, v_s, scale)

    # Reference path: dequantize the SAME k_q/v_q back to fp16, then run the
    # fp32 reference attention. This isolates "INT8 attention kernel
    # correctness" from "quantization accuracy" — both paths see the same
    # quantized inputs.
    k_dq = (k_q.float() * k_s.float().unsqueeze(-1)).to(torch.float16)
    v_dq = (v_q.float() * v_s.float().unsqueeze(-1)).to(torch.float16)
    out_ref = decode_attention(q.unsqueeze(2), k_dq, v_dq, scale=scale).squeeze(2)

    # The two paths differ only in how the dequantization is folded into the
    # arithmetic (in-loop vs upfront materialise-to-fp16). Tolerance is the
    # same as the fp16 attention test: 2e-2 rtol/atol.
    torch.testing.assert_close(out_cuda.float(), out_ref.float(),
                               rtol=2e-2, atol=2e-2)


# ---- INT4 KIVI: CUDA kernels vs PyTorch reference ----

def _unpack_int4_packed(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of the CUDA packing: byte = (q_lo & 0xF) | ((q_hi & 0xF) << 4).

    packed: [..., head_dim/2] int8
    returns: [..., head_dim] int8 with each nibble sign-extended into int8.
    """
    *prefix, half_d = packed.shape
    p = packed.to(torch.int32) & 0xFF  # treat as unsigned for the bit work

    # Low nibble: shift to high bits then arithmetic shift back (sign-extend).
    lo = ((p & 0x0F) << 4).to(torch.int8)
    lo = lo >> 4

    # High nibble: same trick.
    hi = (p & 0xF0).to(torch.int8)
    hi = hi >> 4

    # Interleave: out[..., 2k] = lo, out[..., 2k+1] = hi.
    out = torch.stack([lo, hi], dim=-1).reshape(*prefix, 2 * half_d)
    return out.to(torch.int8)


@requires_cuda
@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024)])
@pytest.mark.parametrize("group_size", [32, 128])
def test_cuda_quantize_k_per_channel_groupwise_int4_matches_reference(
    batch: int, seqlen: int, group_size: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch, N_HEADS_KV, seqlen, HEAD_DIM,
                    device="cuda", dtype=torch.float16)
    # Inject K's per-channel outliers — same pattern as _make_kv on CPU.
    x[..., ::17] *= 3.5

    q_packed_cuda, s_cuda = llmik.quantize_k_per_channel_groupwise_int4(
        x, group_size)
    q_unpacked_cuda = _unpack_int4_packed(q_packed_cuda).to(torch.int8)

    q_ref, s_ref = quantize_per_channel_groupwise(x.float(), bits=4,
                                                  group_size=group_size)
    q_ref = q_ref.to("cuda")
    s_ref = s_ref.to("cuda").to(torch.float16)

    torch.testing.assert_close(s_cuda.float(), s_ref.float(),
                               rtol=1e-3, atol=1e-3)

    diff = (q_unpacked_cuda.int() - q_ref.int()).abs()
    assert (diff <= 1).all(), (
        f"CUDA vs reference int4 K diff > 1 at {(diff > 1).sum().item()} positions"
    )
    assert diff.float().mean().item() < 0.05


@requires_cuda
@pytest.mark.parametrize("batch,seqlen", [(1, 128), (4, 1024), (8, 4096)])
def test_cuda_quantize_v_per_token_int4_matches_reference(
    batch: int, seqlen: int,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(batch, N_HEADS_KV, seqlen, HEAD_DIM,
                    device="cuda", dtype=torch.float16)

    q_packed_cuda, s_cuda = llmik.quantize_v_per_token_int4(x)
    q_unpacked_cuda = _unpack_int4_packed(q_packed_cuda).to(torch.int8)

    q_ref, s_ref = quantize_per_token(x.float(), bits=4)
    q_ref = q_ref.to("cuda")
    s_ref = s_ref.to("cuda").to(torch.float16)

    torch.testing.assert_close(s_cuda.float(), s_ref.float(),
                               rtol=1e-3, atol=1e-3)

    diff = (q_unpacked_cuda.int() - q_ref.int()).abs()
    assert (diff <= 1).all()
    assert diff.float().mean().item() < 0.05


@requires_cuda
@pytest.mark.parametrize("batch,seqlen_kv", [(1, 128), (4, 1024), (8, 2048)])
def test_cuda_decode_attention_int4_matches_dequant_reference(
    batch: int, seqlen_kv: int,
) -> None:
    """End-to-end KIVI: CUDA INT4 attention vs reference attention on the
    dequantized K/V. Isolates kernel correctness from quantization noise."""
    torch.manual_seed(0)
    n_heads, n_kv_heads, head_dim = 32, N_HEADS_KV, HEAD_DIM
    group_size = 32
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, n_heads,    head_dim,
                    device="cuda", dtype=torch.float16)
    k = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                    device="cuda", dtype=torch.float16)
    v = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                    device="cuda", dtype=torch.float16)

    # CUDA quantize → CUDA INT4 attention.
    k_q, k_s = llmik.quantize_k_per_channel_groupwise_int4(k, group_size)
    v_q, v_s = llmik.quantize_v_per_token_int4(v)
    out_cuda = llmik.decode_attention_int4(q, k_q, k_s, v_q, v_s,
                                           group_size, scale)

    # Reference: unpack CUDA's packed output back to int8, dequantize via the
    # per-channel-groupwise / per-token reference, run fp32 attention.
    k_q_unpacked = _unpack_int4_packed(k_q)
    v_q_unpacked = _unpack_int4_packed(v_q)
    k_dq = dequantize_per_channel_groupwise(k_q_unpacked, k_s, group_size).to(torch.float16)
    v_dq = dequantize_per_token(v_q_unpacked, v_s).to(torch.float16)
    out_ref = decode_attention(q.unsqueeze(2), k_dq, v_dq, scale=scale).squeeze(2)

    # INT4 has 16x coarser quantization than INT8 (qmax=7 vs 127), so we
    # need looser tolerance than the INT8 path. Errors come from fp16
    # accumulation only — both paths see the exact same dequantized inputs.
    torch.testing.assert_close(out_cuda.float(), out_ref.float(),
                               rtol=5e-2, atol=5e-2)
