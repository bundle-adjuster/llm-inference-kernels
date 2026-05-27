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
from typing import Iterator, Optional

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


# =============================================================================
# Phase 4b prefill-side patch: KIVI quantize-and-dequantize on F.sdpa inputs.
# =============================================================================
#
# Companion to the decode-side cache replacement below. Prefill-only evals
# (WikiText-2 PPL via `model(input_ids)`, all the lm-evaluation-harness
# log-likelihood tasks) never enter `generate()` and so never touch our
# Int4KIVICache. To make these evals reflect the KIVI accuracy hit, we
# repeat the trick from Phase 2d's `scripts/eval_perplexity.py`: rebind
# `F.scaled_dot_product_attention` so K and V are quantized-then-dequantized
# right before SDPA runs. The original SDPA sees noisy fp16 K/V — same noise
# pattern the model would see if reading from a real INT4 cache.


def _make_kivi_roundtrip_sdpa(n_kv_heads: int, group_size: int):
    """Return a SDPA wrapper that quantize-and-dequantizes K/V using the KIVI
    scheme (per-channel groupwise K, per-token V), then calls the original
    SDPA. Mirrors the math of `Int4KIVICache.update()` so this prefill path's
    noise matches what `decode_attention_int4` reads at decode."""
    from integration.kv_int4_cache import (
        _dequantize_per_channel_groupwise,
        _dequantize_per_token,
        _unpack_int4_packed,
    )

    def patched_sdpa(query, key, value, *args, **kwargs):
        # query/key/value: [B, n_heads, kv_len, D]  (K/V already through repeat_kv)
        if query.dim() != 4 or key.dim() != 4:
            return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)
        bsz, n_heads, _, head_dim = query.shape
        n_rep = n_heads // n_kv_heads
        if n_rep < 1:
            return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

        # Un-expand to GQA shape for the quantize, then re-expand after
        k_unexp = key[:, ::n_rep].contiguous()
        v_unexp = value[:, ::n_rep].contiguous()

        k_packed, k_scale = llmik_cuda.quantize_k_per_channel_groupwise_int4(
            k_unexp, group_size)
        v_packed, v_scale = llmik_cuda.quantize_v_per_token_int4(v_unexp)

        # Round-trip back to fp16 so the original SDPA path can use it
        k_unpacked = _unpack_int4_packed(k_packed)
        v_unpacked = _unpack_int4_packed(v_packed)
        k_noisy = _dequantize_per_channel_groupwise(
            k_unpacked, k_scale, group_size)
        v_noisy = _dequantize_per_token(v_unpacked, v_scale)

        # Re-expand to match query's head count (repeat_kv equivalent)
        k_noisy = k_noisy.repeat_interleave(n_rep, dim=1)
        v_noisy = v_noisy.repeat_interleave(n_rep, dim=1)

        return _ORIGINAL_SDPA(query, k_noisy, v_noisy, *args, **kwargs)

    return patched_sdpa


@contextmanager
def patched_kivi_int4_sdpa(n_kv_heads: int, group_size: int) -> Iterator[None]:
    """Rebind F.sdpa to apply KIVI quantize-dequantize on K/V before calling
    the original SDPA. Use for prefill-only evals (PPL, lm-eval-harness)."""
    patched = _make_kivi_roundtrip_sdpa(n_kv_heads, group_size)
    F.scaled_dot_product_attention = patched
    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA


# =============================================================================
# Phase 4b: LlamaSdpaAttention.forward replacement for the INT4 KIVI cache.
# =============================================================================
#
# We replace the whole forward at decode-time (q_len == 1) so we can:
#   - bypass HF's `repeat_kv` (saves bandwidth)
#   - bypass `F.scaled_dot_product_attention` (which doesn't know how to read
#     packed-int4 K/V)
#   - call our `decode_attention_int4` kernel directly on the cache's stored
#     packed tensors (no dequant-then-requant waste)
#
# Prefill (q_len > 1) still flows through the original forward, where our
# Int4KIVICache.update() returns dequantized fp16 for the original SDPA call.


def _make_int4_decode_forward(original_forward, group_size: int):
    """Build the replacement LlamaSdpaAttention.forward. Closes over the
    cache's `group_size` so the kernel call gets it right."""
    # Late import to avoid loading transformers' modeling_llama at module
    # import time (some import-order paths trip the warnings system).
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from integration.kv_int4_cache import Int4KIVICache

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # Only handle the decode + INT4 case here; everything else falls back.
        is_decode = (hidden_states.dim() == 3 and hidden_states.size(1) == 1)
        if (not is_decode
                or not isinstance(past_key_value, Int4KIVICache)
                or output_attentions
                or position_embeddings is None):
            return original_forward(
                self, hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # === INT4 decode fast path ===
        bsz, q_len, _ = hidden_states.size()  # q_len == 1

        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [B, n_heads, 1, D]
        k = k.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [B, n_kv, 1, D]
        v = v.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Append the new token to the cache; skips the fp16 materialize since
        # we read packed tensors via get_quantized_for_attention() below.
        past_key_value.append_only(k.contiguous(), v.contiguous(), self.layer_idx)

        k_packed, k_scale, v_packed, v_scale = (
            past_key_value.get_quantized_for_attention(self.layer_idx))

        # Our int4 decode kernel
        q_flat = q.squeeze(2).contiguous()  # [B, n_heads, D]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = llmik_cuda.decode_attention_int4(
            q_flat, k_packed, k_scale, v_packed, v_scale,
            group_size, scale)
        attn_out = attn_out.unsqueeze(2)  # [B, n_heads, 1, D]

        # Output projection
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
        attn_out = self.o_proj(attn_out)

        return attn_out, None, past_key_value

    return forward


@contextmanager
def patched_int4_decode_attention(group_size: int) -> Iterator[None]:
    """Replace LlamaSdpaAttention.forward to call our INT4 decode kernel when
    the cache is an `Int4KIVICache` and `q_len == 1`. Prefill and other
    contexts fall through to the original forward (which uses the cache's
    fp16-materialize path).

    Args:
        group_size: KIVI group size for K quantization; must match the cache.
    """
    from transformers.models.llama.modeling_llama import LlamaSdpaAttention

    original_forward = LlamaSdpaAttention.forward
    new_forward = _make_int4_decode_forward(original_forward, group_size)
    LlamaSdpaAttention.forward = new_forward
    try:
        yield
    finally:
        LlamaSdpaAttention.forward = original_forward
