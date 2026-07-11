"""Cat-free decode path: a preallocated KV cache + the v6 length-aware kernel.

The Phase 7 attribution found that transformers' `DynamicCache` re-`torch.cat`s the
whole KV cache every decode step (~2 ms/step at batch=16), and that the obvious
fix — a `StaticCache` — drops PyTorch SDPA into its math backend (the mask
disqualifies flash), making it *slower*. So a static cache only pays off with an
attention kernel that takes a live length and reads a contiguous prefix in place.

That is exactly what the Phase 8 v6 kernel became when it gained a `seqlen`
argument (Phase 9): `PreallocCache` writes each new token in place into a
`[B, n_kv, max_len, D]` buffer (no cat) and hands the *full* buffer to v6, which
attends over `[0, live_len)` directly. Greedy output is bit-identical to the
`DynamicCache` path; the step just loses the cat.

Usage:
    cache = PreallocCache(max_len=prompt_len + gen_len)
    with v6_decode(cache):
        model(input_ids=..., past_key_values=cache, use_cache=True, cache_position=...)
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama import modeling_llama as _ml

import llmik_cuda

_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, kvh, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


class PreallocCache(DynamicCache):
    """KV cache that writes decode tokens in place — no per-step `torch.cat`.

    Prefill stores its K/V in the preallocated buffer and returns the contiguous
    prefill tensors (attention takes the stock expanded-SDPA path). Decode writes
    the single new token at the live offset and returns the *full* buffer; the v6
    hook reads its `[0, cur_attn_len)` prefix. Per-layer buffers are allocated
    lazily on first use. `cur_attn_len` is the length the immediately-following
    attention call should attend over (set by the update that precedes it).
    """

    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.max_len = max_len
        self.kbuf: dict[int, torch.Tensor] = {}
        self.vbuf: dict[int, torch.Tensor] = {}
        self.lengths: dict[int, int] = {}
        self.cur_attn_len = 0

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx not in self.kbuf:
            b, h, _, d = key.shape
            self.kbuf[layer_idx] = key.new_empty((b, h, self.max_len, d))
            self.vbuf[layer_idx] = value.new_empty((b, h, self.max_len, d))
            self.lengths[layer_idx] = 0
        s = key.shape[2]
        pos = self.lengths[layer_idx]
        self.kbuf[layer_idx][:, :, pos:pos + s, :] = key
        self.vbuf[layer_idx][:, :, pos:pos + s, :] = value
        self.lengths[layer_idx] = pos + s
        self.cur_attn_len = pos + s
        if s > 1:                                   # prefill: contiguous tensors
            return key, value
        return self.kbuf[layer_idx], self.vbuf[layer_idx]   # decode: full buffer

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.lengths.get(layer_idx, 0)


@contextlib.contextmanager
def v6_decode(cache: PreallocCache):
    """Route decode attention (q_len==1) to the v6 kernel over the cache's live
    prefix, with `repeat_kv` neutralized. Prefill keeps the stock expanded SDPA
    path (byte-identical across configs). Restores globals on exit."""
    orig_repeat = _ml.repeat_kv

    def identity(hidden_states, n_rep):
        return hidden_states

    def hook(query, key, value, *args, **kwargs):
        if query.size(2) > 1:                       # prefill
            n_rep = query.size(1) // key.size(1)
            return _ORIGINAL_SDPA(query, _repeat_kv(key, n_rep),
                                  _repeat_kv(value, n_rep), *args, **kwargs)
        scale = 1.0 / math.sqrt(query.size(-1))     # decode: v6 over [0, live_len)
        out = llmik_cuda.decode_attention(
            query.squeeze(2).contiguous(), key, value, scale, cache.cur_attn_len)
        return out.unsqueeze(2)

    _ml.repeat_kv = identity
    F.scaled_dot_product_attention = hook
    torch.nn.functional.scaled_dot_product_attention = hook
    try:
        yield
    finally:
        _ml.repeat_kv = orig_repeat
        F.scaled_dot_product_attention = _ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA


@contextlib.contextmanager
def fused_decode(cache: PreallocCache):
    """`v6_decode` plus fused RMSNorm and SwiGLU (Phase 10).

    transformers runs RMSNorm as ~5 kernels (pow/mean/rsqrt/mul/mul) and SwiGLU
    as two (silu, then mul); replacing each with one fused kernel is what a
    serving engine does. This is what closes the last fp16-vs-fp16 gap to vLLM
    (the projection GEMMs already go through cuBLAS, same as vLLM). Both fused
    ops match the reference within fp16 rounding; greedy output is preserved.
    """
    orig_rms = _ml.LlamaRMSNorm.forward
    orig_mlp = _ml.LlamaMLP.forward
    orig_rope = _ml.apply_rotary_pos_emb

    def rms_fwd(self, hidden_states):
        return llmik_cuda.rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    def mlp_fwd(self, x):
        return self.down_proj(llmik_cuda.silu_mul(self.gate_proj(x), self.up_proj(x)))

    def rope_fwd(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        return llmik_cuda.rope(q, cos, sin), llmik_cuda.rope(k, cos, sin)

    _ml.LlamaRMSNorm.forward = rms_fwd
    _ml.LlamaMLP.forward = mlp_fwd
    _ml.apply_rotary_pos_emb = rope_fwd
    try:
        with v6_decode(cache):
            yield
    finally:
        _ml.LlamaRMSNorm.forward = orig_rms
        _ml.LlamaMLP.forward = orig_mlp
        _ml.apply_rotary_pos_emb = orig_rope
