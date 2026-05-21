"""PyTorch reference implementations — the correctness oracles.

Intentionally simple and obviously correct. Custom CUDA kernels are validated
against these (within FP16 tolerance) before any performance number is taken.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _expand_gqa(k, v, n_heads):
    """Repeat KV heads so they match the query head count (GQA)."""
    n_kv_heads = k.shape[1]
    if n_kv_heads != n_heads:
        rep = n_heads // n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return k, v


def eager_attention(q, k, v, *, scale=None, causal=False):
    """Naive attention that materializes the score matrix. Obviously correct.

    q:    [batch, n_heads, q_len, head_dim]
    k, v: [batch, n_kv_heads, kv_len, head_dim]   (GQA: n_kv_heads <= n_heads)
    -> [batch, n_heads, q_len, head_dim]
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


def sdpa_attention(q, k, v, *, causal=False):
    """PyTorch SDPA — dispatches to FlashAttention / cuDNN. A SOTA baseline."""
    k, v = _expand_gqa(k, v, q.shape[1])
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def decode_attention(q, k, v, *, scale=None):
    """Decode case: a single query token. q: [batch, n_heads, 1, head_dim]."""
    return eager_attention(q, k, v, scale=scale, causal=False)
