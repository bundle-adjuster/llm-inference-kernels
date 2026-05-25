"""W4A16 quantized matmul reference — PyTorch correctness oracle.

Design: docs/03-quantized-matmul.md.

Symmetric per-channel groupwise weight quantization at 4 bits:
  - Weights are `[K, N]` row-major (K = in_features, N = out_features),
    matched to the GEMM `out[M, N] = act[M, K] @ W[K, N]`.
  - Scales are groupwise along K with `group_size` (typically 128). One
    fp16 scale per (group along K, output channel along N).
  - 4-bit values land in [-7, 7]. Reference stores them in int8
    containers (unpacked) for clarity; the CUDA kernel in Phase 3b
    packs into uint32 along K.

Quantize:
    scale[g, n] = max(|W[g·group_size : (g+1)·group_size, n]|) / 7
    q[k, n]     = round(W[k, n] / scale[k // group_size, n]),
                  clamped to [-7, 7].

Dequantize:
    W_hat[k, n] = q[k, n] · scale[k // group_size, n]

Matmul (the GEMM the kernel computes):
    out[m, n] = Σ_k act[m, k] · W_hat[k, n]

This file is the contract every Phase 3 CUDA kernel must match. Same
pattern as `reference/kv_cache_ref.py` in Phase 2.
"""
from __future__ import annotations

from typing import Tuple

import torch


# 4-bit symmetric range: [-7, 7].
_QMAX_INT4: int = 7


def quantize_weights_int4_groupwise(
    W: torch.Tensor, group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 groupwise quantization of a weight matrix.

    Args:
        W: `[K, N]` weight matrix in fp16 or fp32. Layout: K is the
            reduction (in_features) axis; N is the output (out_features)
            axis. Note this is the *transpose* of `torch.nn.Linear.weight`
            (which is `[N, K]`); pass `W.T.contiguous()` to use a Linear
            layer's weights.
        group_size: number of K positions per scale (typically 128).
            K must be a multiple of `group_size` for the kernel
            (the reference handles trailing partial groups with zero-pad
            during scale computation, but the kernel won't).

    Returns:
        q: `[K, N]` int8 with values in [-7, 7] (unpacked — one nibble
            per byte). The CUDA kernel will pack 8 nibbles per uint32
            along K; the reference doesn't pack, for clarity.
        scale: `[n_groups, N]` in the same dtype as W. One fp16 scale per
            (group_along_K, output_channel).
    """
    K, N = W.shape
    n_groups = (K + group_size - 1) // group_size
    pad = n_groups * group_size - K

    if pad > 0:
        # Pad K with zeros so we can reshape cleanly. Zeros don't move
        # absmax up, so they don't disturb the scale of the short final group.
        W_padded = torch.cat(
            [W, torch.zeros(pad, N, device=W.device, dtype=W.dtype)],
            dim=0,
        )
    else:
        W_padded = W

    # Reshape K -> (n_groups, group_size) so the absmax reduction along
    # the in-group axis lands one scale per (group, output channel).
    W_grp = W_padded.view(n_groups, group_size, N)
    absmax = W_grp.abs().amax(dim=1)            # [n_groups, N]
    scale = (absmax / _QMAX_INT4).clamp(min=1e-8)

    # Broadcast scale across the in-group axis for elementwise quantize.
    q_grp = (W_grp.float() / scale.unsqueeze(1)).round().clamp(
        -_QMAX_INT4, _QMAX_INT4,
    ).to(torch.int8)
    q = q_grp.view(n_groups * group_size, N)[:K, :]

    return q, scale.to(W.dtype)


def dequantize_weights_int4_groupwise(
    q: torch.Tensor, scale: torch.Tensor, group_size: int,
) -> torch.Tensor:
    """Invert :func:`quantize_weights_int4_groupwise`.

    Args:
        q: `[K, N]` int8 (values in [-7, 7]).
        scale: `[n_groups, N]` in any fp dtype.
        group_size: must match the quantize-time value.

    Returns:
        `[K, N]` fp32 (cast at the call site if needed).
    """
    K, _ = q.shape
    # For each k in [0, K), look up the group it belongs to.
    group_idx = torch.arange(K, device=q.device) // group_size  # [K]
    # scale.index_select(0, group_idx) -> [K, N], scale-per-row matching q.
    scale_expanded = scale.index_select(dim=0, index=group_idx)
    return q.float() * scale_expanded.float()


def quantized_matmul_ref(
    act: torch.Tensor,
    q_weights: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Reference W4A16 matmul: `out = act @ dequant(W)`.

    The CUDA kernel in Phase 3b/3c fuses the dequantization into the
    inner reduction (no fp16 W ever materialised); the reference
    materialises it for clarity.

    Args:
        act: `[M, K]` fp16 activations.
        q_weights: `[K, N]` int8 (unpacked int4 in [-7, 7]).
        scale: `[n_groups, N]` fp16, one scale per (group along K, N).
        group_size: tokens-along-K per scale group.

    Returns:
        `[M, N]` in `act.dtype`. Internal compute in fp32 so the
        reference is trustworthy even at fp16 input.
    """
    W_hat = dequantize_weights_int4_groupwise(q_weights, scale, group_size)
    return (act.float() @ W_hat).to(act.dtype)


# ---- Convenience preset: end-to-end one-liner ----

def pack_int4_along_k(q_int8: torch.Tensor) -> torch.Tensor:
    """Pack `[K, N] int8` weights (values in [-7, 7]) into `[K/8, N] int32`.

    Each output int32 holds 8 K-values for one output column. Bit
    `i*4 .. i*4+3` stores K position `k_pack*8 + i`. The kernel reads
    one int32 per inner-loop iter and unpacks 8 nibbles via the
    shift-trick `(signed_w << (28 - i*4)) >> 28` (arithmetic right
    shift sign-extends in PTX).

    Args:
        q_int8: `[K, N]` int8 with values in [-7, 7]. K must be a
            multiple of 8.

    Returns:
        `[K/8, N]` int32. Torch has no native uint32; the int32 bit
        pattern is identical to uint32 and the CUDA kernel
        reinterprets it as such.
    """
    K, N = q_int8.shape
    assert K % 8 == 0, f"K={K} must be a multiple of 8 to pack int4 along K"
    # q_grp[k_pack, i, n] = the i-th K-value in the k_pack-th 8-block
    q_grp = q_int8.view(K // 8, 8, N).to(torch.int32) & 0xF
    packed = torch.zeros(K // 8, N, dtype=torch.int32, device=q_int8.device)
    for i in range(8):
        packed = packed | (q_grp[:, i, :] << (i * 4))
    return packed


def w4a16_matmul_ref(
    act: torch.Tensor, W: torch.Tensor, group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize W on the fly and return (out, q, scale).

    Useful for tests that want to compare `out_quantized` against the
    fp16 reference `act @ W` and see the quantization noise floor.

    Args:
        act: `[M, K]`.
        W: `[K, N]` fp16.
        group_size: defaults to 128 (the docs/03 baseline).

    Returns:
        out: `[M, N]` in `act.dtype` — the quantized-matmul output.
        q: `[K, N]` int8.
        scale: `[n_groups, N]` in `W.dtype`.
    """
    q, scale = quantize_weights_int4_groupwise(W, group_size=group_size)
    out = quantized_matmul_ref(act, q, scale, group_size)
    return out, q, scale
