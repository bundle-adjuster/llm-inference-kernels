"""KV-cache quantization reference — PyTorch correctness oracle.

Design: docs/02-kv-cache-compression.md.

Symmetric integer quantization. Two axis modes:

- **Per-token**: one scale per `[batch, n_kv_heads, seqlen]` position, shared
  across `head_dim`. Natural for V (no persistent per-channel outliers).

- **Per-channel groupwise**: one scale per `[batch, n_kv_heads, n_groups,
  head_dim]` cell, where groups partition the seqlen axis (group_size tokens
  per group). Natural for K — KIVI keeps INT4-K accuracy by absorbing K's
  per-channel outliers into per-channel scales, with groups giving spatial
  locality so scales adapt as the cache streams.

KIVI defaults:
  K: bits=4, axis="channel", group_size=32
  V: bits=4, axis="token",   group_size=None

INT8 baseline (simpler, near-lossless):
  K, V: bits=8, axis="token", group_size=None

The CUDA kernels in Phase 2b/c are validated against these references.
"""
from __future__ import annotations

from typing import Tuple

import torch


def _qmax(bits: int) -> int:
    """Symmetric integer-range upper bound: 2^(bits-1) − 1.

    Args:
        bits: 4 or 8.

    Returns:
        7 for 4-bit, 127 for 8-bit. Quantized values land in [-qmax, qmax]
        (i.e. -127..127 or -7..7), never the asymmetric INT_MIN.
    """
    assert bits in (4, 8), f"only int4/int8 supported, got bits={bits}"
    return (1 << (bits - 1)) - 1


def quantize_per_token(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-token quantization.

    x:     [batch, n_kv_heads, seqlen, head_dim]
    bits:  4 or 8

    Returns:
      q:     [batch, n_kv_heads, seqlen, head_dim]  int8  (values in
             [-qmax, qmax]; for bits=4 the int8 container holds [-7, 7])
      scale: [batch, n_kv_heads, seqlen]            same dtype as x
             (one scale per token, broadcast across head_dim on dequant)
    """
    qmax = _qmax(bits)
    absmax = x.abs().amax(dim=-1, keepdim=True)              # [..., seqlen, 1]
    scale = (absmax / qmax).clamp(min=1e-8)
    q = (x.float() / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q, scale.squeeze(-1).to(x.dtype)


def dequantize_per_token(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quantize_per_token`.

    Returns an fp32 tensor; cast at the call site if needed.
    """
    return q.float() * scale.float().unsqueeze(-1)


def quantize_per_channel_groupwise(
    x: torch.Tensor, bits: int, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel quantization with groups along the seqlen axis.

    Each (batch, kv_head, group_of_tokens, head_dim_channel) gets its own scale.

    x:          [batch, n_kv_heads, seqlen, head_dim]
    bits:       4 or 8
    group_size: tokens per group along the seqlen axis

    Returns:
      q:     [batch, n_kv_heads, seqlen, head_dim]  int8
      scale: [batch, n_kv_heads, n_groups, head_dim]  where
             n_groups = ceil(seqlen / group_size)
    """
    qmax = _qmax(bits)
    batch, n_kv, seqlen, head_dim = x.shape
    n_groups = (seqlen + group_size - 1) // group_size
    pad = n_groups * group_size - seqlen

    if pad > 0:
        # Pad with zeros — they don't change abs-max within the (short) final group.
        x_padded = torch.cat(
            [x, torch.zeros(batch, n_kv, pad, head_dim,
                            device=x.device, dtype=x.dtype)],
            dim=2,
        )
    else:
        x_padded = x

    x_grp = x_padded.view(batch, n_kv, n_groups, group_size, head_dim)
    # Reduce across the in-group seqlen axis (dim=3) → per (batch, kv, group, channel).
    absmax = x_grp.abs().amax(dim=3)                          # [b, kv, ng, d]
    scale = (absmax / qmax).clamp(min=1e-8)

    # Broadcast scale to per-token-per-channel: [b, kv, ng, 1, d].
    q_grp = (x_grp.float() / scale.unsqueeze(3)).round().clamp(-qmax, qmax).to(torch.int8)
    q = q_grp.view(batch, n_kv, n_groups * group_size, head_dim)[:, :, :seqlen, :]
    return q, scale.to(x.dtype)


def dequantize_per_channel_groupwise(
    q: torch.Tensor, scale: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of :func:`quantize_per_channel_groupwise`.

    q:          [batch, n_kv_heads, seqlen, head_dim]
    scale:      [batch, n_kv_heads, n_groups, head_dim]
    group_size: must match what was used at quantization time
    """
    batch, n_kv, seqlen, head_dim = q.shape
    # Map each token index to its group index, then gather along dim 2.
    token_to_group = torch.arange(seqlen, device=q.device) // group_size
    # scale[:, :, token_to_group, :] → [batch, n_kv, seqlen, head_dim]
    scale_expanded = scale.index_select(dim=2, index=token_to_group)
    return q.float() * scale_expanded.float()


# Convenience: KIVI-style and INT8-flat presets that take a full
# (K, V) pair and return packed plus scales. These are what Phase 2b/c
# CUDA kernels will mirror.

def quantize_kv_int8_per_token(
    k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT8 baseline: both K and V quantized per-token.

    Returns (k_q, k_scale, v_q, v_scale).
    """
    k_q, k_s = quantize_per_token(k, bits=8)
    v_q, v_s = quantize_per_token(v, bits=8)
    return k_q, k_s, v_q, v_s


def dequantize_kv_int8_per_token(
    k_q: torch.Tensor, k_scale: torch.Tensor,
    v_q: torch.Tensor, v_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = dequantize_per_token(k_q, k_scale).to(out_dtype)
    v = dequantize_per_token(v_q, v_scale).to(out_dtype)
    return k, v


def quantize_kv_kivi_int4(
    k: torch.Tensor, v: torch.Tensor, group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KIVI INT4: K per-channel groupwise; V per-token.

    Returns (k_q, k_scale, v_q, v_scale). The int8 container holds 4-bit
    values in [-7, 7]; an actual storage path would pack two nibbles per
    byte — that lives in the CUDA kernel.
    """
    k_q, k_s = quantize_per_channel_groupwise(k, bits=4, group_size=group_size)
    v_q, v_s = quantize_per_token(v, bits=4)
    return k_q, k_s, v_q, v_s


def dequantize_kv_kivi_int4(
    k_q: torch.Tensor, k_scale: torch.Tensor,
    v_q: torch.Tensor, v_scale: torch.Tensor,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = dequantize_per_channel_groupwise(k_q, k_scale, group_size).to(out_dtype)
    v = dequantize_per_token(v_q, v_scale).to(out_dtype)
    return k, v
