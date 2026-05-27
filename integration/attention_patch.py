"""Phase 4a: plumb the Phase 1 v3 decode_attention kernel into HF Llama.

Strategy: rebind `torch.nn.functional.scaled_dot_product_attention` (same hook
used by `scripts/eval_perplexity.py` in Phase 2d). HF's `LlamaSdpaAttention.forward`
looks the symbol up *on the module* at call time, so a rebind intercepts every
layer's attention with no class subclassing required.

The rebind dispatches:

  q_len == 1   →  our `llmik_cuda.decode_attention` kernel  (Phase 1 v3)
  q_len  > 1   →  the original `F.scaled_dot_product_attention` (prefill)

Our kernel takes K/V in the original GQA shape `[B, n_kv_heads, kv_len, D]`,
but HF runs `repeat_kv` BEFORE the SDPA call, expanding to
`[B, n_heads, kv_len, D]`. We undo that with `K[:, ::n_rep]` — repeat_kv is a
`repeat_interleave` along dim 1, so taking every n_rep-th head recovers the
original. Without this un-expansion the kernel would read 4× the KV bandwidth
on Llama 3.1 8B (n_heads=32, n_kv_heads=8) and lose its whole point.

The slice `K[:, ::n_rep]` is non-contiguous; `.contiguous()` copies the
un-expanded KV into a fresh buffer per call. For batch=16, kv_len=1024, that's
~33 MB at HBM speeds ≈ 33 µs per decode step — small relative to our kernel's
runtime, fine for the first integration.

Usage:
    from integration.attention_patch import patched_decode_attention
    with patched_decode_attention(n_kv_heads=8):
        out = model.generate(...)
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

import llmik_cuda

_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _make_decode_sdpa(n_kv_heads: int):
    """Build the patched SDPA callable. Closes over `n_kv_heads` so the slice
    factor `n_rep = n_heads // n_kv_heads` can be computed once per call."""

    def patched_sdpa(query, key, value, *args, **kwargs):
        # query: [B, n_heads, q_len, head_dim]
        # key/value: [B, n_heads, kv_len, head_dim]  (already through repeat_kv)
        if query.dim() != 4 or query.size(2) != 1:
            return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

        bsz, n_heads, _, head_dim = query.shape
        n_rep = n_heads // n_kv_heads
        if n_rep < 1:
            # Defensive: caller passed wrong n_kv_heads.
            return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

        # Un-expand the GQA repeat_kv: take heads 0, n_rep, 2*n_rep, ...
        k_unexp = key[:, ::n_rep].contiguous()
        v_unexp = value[:, ::n_rep].contiguous()

        q_flat = query.squeeze(2).contiguous()  # [B, n_heads, head_dim]
        scale = 1.0 / math.sqrt(head_dim)
        out = llmik_cuda.decode_attention(q_flat, k_unexp, v_unexp, scale)
        return out.unsqueeze(2)  # back to [B, n_heads, 1, head_dim]

    return patched_sdpa


@contextmanager
def patched_decode_attention(n_kv_heads: int) -> Iterator[None]:
    """Rebind `F.scaled_dot_product_attention` to our decode kernel for the
    duration of the `with` block. Restores the original on exit (even on error).

    Args:
        n_kv_heads: the model's original `num_key_value_heads` *before*
            HF's `repeat_kv` expansion. For Llama 3.1 8B Instruct this is 8.
    """
    patched = _make_decode_sdpa(n_kv_heads)
    F.scaled_dot_product_attention = patched
    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
