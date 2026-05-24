"""PyTorch reference implementations — the correctness oracles.

Intentionally simple and obviously correct. Custom CUDA kernels are validated
against these (within FP16 tolerance) before any performance number is taken.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _expand_gqa(
    k: torch.Tensor, v: torch.Tensor, n_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads so they match the query head count (GQA expansion).

    Llama 3 8B has `n_heads=32` query heads but only `n_kv_heads=8` KV
    heads — each KV head is shared by `n_heads / n_kv_heads = 4` query
    heads. The custom CUDA kernels handle this with an index map; the
    reference just `repeat_interleave`s for simplicity.

    Args:
        k: `[batch, n_kv_heads, kv_len, head_dim]`.
        v: `[batch, n_kv_heads, kv_len, head_dim]`.
        n_heads: target number of query heads.

    Returns:
        (k_expanded, v_expanded) with shape
        `[batch, n_heads, kv_len, head_dim]`. If `n_kv_heads == n_heads`,
        returns the inputs unchanged.
    """
    n_kv_heads = k.shape[1]
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def eager_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Naive attention that materializes the score matrix. Obviously correct.

    Computed in fp32 internally regardless of input dtype, so this is a
    trustworthy reference even for fp16 inputs.

    Args:
        q: `[batch, n_heads, q_len, head_dim]`.
        k: `[batch, n_kv_heads, kv_len, head_dim]` (GQA expansion handled).
        v: `[batch, n_kv_heads, kv_len, head_dim]`.
        scale: softmax scale; defaults to `1/sqrt(head_dim)` if None.
        causal: if True, apply a (q_len × kv_len) causal mask aligned to
            the right edge (so `q[i]` attends to `k[0..kv_len - q_len + i]`).

    Returns:
        `[batch, n_heads, q_len, head_dim]` in `q.dtype`.
    """
    batch, n_heads, q_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    k, v = _expand_gqa(k, v, n_heads)

    # Compute in fp32 for a trustworthy reference regardless of input dtype.
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        kv_len = k.shape[2]
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def sdpa_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch SDPA — dispatches to FlashAttention / cuDNN. A SOTA baseline.

    Same input/output contract as `eager_attention` but goes through
    `F.scaled_dot_product_attention`, which on Ada GPUs dispatches to
    FlashAttention-2 or cuDNN's flash-attention backend.
    """
    k, v = _expand_gqa(k, v, q.shape[1])
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Decode-time reference: a single query token attends to a KV cache.

    Args:
        q: `[batch, n_heads, 1, head_dim]` (one query position).
        k, v: `[batch, n_kv_heads, kv_len, head_dim]`.
        scale: softmax scale; defaults to `1/sqrt(head_dim)`.

    Returns:
        `[batch, n_heads, 1, head_dim]` in `q.dtype`.
    """
    return eager_attention(q, k, v, scale=scale, causal=False)
