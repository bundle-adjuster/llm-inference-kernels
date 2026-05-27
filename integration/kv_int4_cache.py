"""Phase 4b: HF Cache subclass that stores K/V as INT4 KIVI-packed.

KIVI: per-channel groupwise K (group_size=32 along seqlen) + per-token V,
packed 4-bit. Built on top of the Phase 2c CUDA kernels for quantize/dequantize.

Storage strategy:

  Quantized portion : K packed [B, n_kv, n_q, d/2] int8
                      K scales  [B, n_kv, n_groups, d] fp16
                      V packed  [B, n_kv, n_q, d/2] int8
                      V scales  [B, n_kv, n_q] fp16
  Residual portion  : fp16 buffer of recent tokens not yet in a full group
                      (at most `group_size - 1` tokens per layer)

When `r_len >= group_size`, the residual's leading `group_size` tokens are
quantized and moved into the quantized portion. This preserves KIVI's
per-channel-K design: the scale is computed from `group_size` tokens'
worth of channel values, not from a single token.

`update(K, V, layer_idx)` is the standard HF Cache API: it appends new
tokens and returns dequantized fp16 K/V (for SDPA path compatibility,
prefill case). `get_quantized_for_attention(layer_idx)` is our extension:
returns packed INT4 tensors covering all stored tokens (residual quantized
on-the-fly), suitable for our `decode_attention_int4` kernel.

Memory: at full sequence (e.g. kv_len=1024), the cache stores ~0.27x the
bytes of an equivalent fp16 cache (per Phase 2 measurements, dominated by
the int4 payload — scales + residual are small).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache

import llmik_cuda


def _unpack_int4_packed(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of the CUDA packing: byte = (q_lo & 0xF) | ((q_hi & 0xF) << 4).
    Returns int8 tensor with each nibble sign-extended.

    Same logic as `tests/test_kv_cache.py::_unpack_int4_packed`."""
    *prefix, half_d = packed.shape
    p = packed.to(torch.int32) & 0xFF
    lo = ((p & 0x0F) << 4).to(torch.int8) >> 4
    hi = (p & 0xF0).to(torch.int8) >> 4
    out = torch.stack([lo, hi], dim=-1).reshape(*prefix, 2 * half_d)
    return out.to(torch.int8)


def _dequantize_per_channel_groupwise(
    q: torch.Tensor, scale: torch.Tensor, group_size: int,
) -> torch.Tensor:
    """Per-channel groupwise dequantization for K. Same math as
    `reference/kv_cache_ref.py::dequantize_per_channel_groupwise`, returns fp16."""
    seqlen = q.shape[2]
    token_to_group = torch.arange(seqlen, device=q.device) // group_size
    scale_expanded = scale.index_select(dim=2, index=token_to_group)
    return (q.to(torch.float32) * scale_expanded.to(torch.float32)).to(torch.float16)


def _dequantize_per_token(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-token dequantization for V. Returns fp16."""
    return (q.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(torch.float16)


class Int4KIVICache(Cache):
    """KIVI INT4 KV cache. See module docstring for storage layout."""

    def __init__(self, group_size: int = 32, num_layers: int = 32) -> None:
        super().__init__()
        self.group_size = group_size
        self._seen_tokens = 0
        # Quantized storage, one entry per layer (None until first update)
        self.k_packed: List[Optional[torch.Tensor]] = [None] * num_layers
        self.k_scale: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_packed: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_scale: List[Optional[torch.Tensor]] = [None] * num_layers
        # Fp16 residual buffer, one entry per layer (None until first update)
        self.k_resid: List[Optional[torch.Tensor]] = [None] * num_layers
        self.v_resid: List[Optional[torch.Tensor]] = [None] * num_layers

    # ---- HF Cache API ----

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """HF Cache contract: append + return dequantized fp16 K/V for SDPA.
        Use `append_only()` if you don't need the fp16 (e.g. our int4 decode
        fast path) — that skips the expensive whole-cache dequant."""
        self.append_only(key_states, value_states, layer_idx)
        return self._materialize_fp16(layer_idx)

    def append_only(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Store path of `update()` without the fp16 materialize. Drains full
        groups out of the residual into the quantized storage.

        key_states / value_states: [B, n_kv_heads, q_len, head_dim] fp16."""
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Append to residual
        if self.k_resid[layer_idx] is None:
            self.k_resid[layer_idx] = key_states
            self.v_resid[layer_idx] = value_states
        else:
            self.k_resid[layer_idx] = torch.cat(
                [self.k_resid[layer_idx], key_states], dim=2)
            self.v_resid[layer_idx] = torch.cat(
                [self.v_resid[layer_idx], value_states], dim=2)

        # Drain full groups from the residual into quantized storage
        while self.k_resid[layer_idx].shape[2] >= self.group_size:
            chunk_k = self.k_resid[layer_idx][:, :, :self.group_size].contiguous()
            chunk_v = self.v_resid[layer_idx][:, :, :self.group_size].contiguous()
            k_q, k_s = llmik_cuda.quantize_k_per_channel_groupwise_int4(
                chunk_k, self.group_size)
            v_q, v_s = llmik_cuda.quantize_v_per_token_int4(chunk_v)
            if self.k_packed[layer_idx] is None:
                self.k_packed[layer_idx] = k_q
                self.k_scale[layer_idx] = k_s
                self.v_packed[layer_idx] = v_q
                self.v_scale[layer_idx] = v_s
            else:
                self.k_packed[layer_idx] = torch.cat(
                    [self.k_packed[layer_idx], k_q], dim=2)
                self.k_scale[layer_idx] = torch.cat(
                    [self.k_scale[layer_idx], k_s], dim=2)
                self.v_packed[layer_idx] = torch.cat(
                    [self.v_packed[layer_idx], v_q], dim=2)
                self.v_scale[layer_idx] = torch.cat(
                    [self.v_scale[layer_idx], v_s], dim=2)
            # Pop the drained tokens from the residual
            self.k_resid[layer_idx] = self.k_resid[layer_idx][:, :, self.group_size:]
            self.v_resid[layer_idx] = self.v_resid[layer_idx][:, :, self.group_size:]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total tokens in the cache for `layer_idx` (quantized + residual)."""
        n = 0
        if self.k_packed[layer_idx] is not None:
            n += self.k_packed[layer_idx].shape[2]
        if self.k_resid[layer_idx] is not None:
            n += self.k_resid[layer_idx].shape[2]
        return n

    def get_max_cache_shape(self) -> Optional[int]:
        """We grow unboundedly; HF treats `None` as `unbounded`."""
        return None

    def get_max_length(self) -> Optional[int]:
        """Older HF API name for the same thing — kept for compatibility."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Beam search hook. We don't use beam search in Phase 4 eval, so this
        is unimplemented — raise if anyone tries."""
        raise NotImplementedError(
            "Int4KIVICache does not support beam search reordering")

    # ---- Extension methods for our int4 attention kernel ----

    def get_quantized_for_attention(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (k_packed, k_scale, v_packed, v_scale) covering ALL stored
        tokens for this layer. The residual portion (if any) is quantized
        on-the-fly so the kernel sees one contiguous int4 view.

        Quantizing the residual is cheap (at most group_size-1 = 31 tokens) and
        only happens at decode time when we're about to call the kernel."""
        k_packed_q = self.k_packed[layer_idx]
        k_scale_q = self.k_scale[layer_idx]
        v_packed_q = self.v_packed[layer_idx]
        v_scale_q = self.v_scale[layer_idx]
        resid_k = self.k_resid[layer_idx]
        resid_v = self.v_resid[layer_idx]

        if resid_k is None or resid_k.shape[2] == 0:
            return k_packed_q, k_scale_q, v_packed_q, v_scale_q

        # Quantize the residual chunk
        k_res_q, k_res_s = llmik_cuda.quantize_k_per_channel_groupwise_int4(
            resid_k.contiguous(), self.group_size)
        v_res_q, v_res_s = llmik_cuda.quantize_v_per_token_int4(
            resid_v.contiguous())

        if k_packed_q is None:
            return k_res_q, k_res_s, v_res_q, v_res_s

        k_q_all = torch.cat([k_packed_q, k_res_q], dim=2)
        k_s_all = torch.cat([k_scale_q, k_res_s], dim=2)
        v_q_all = torch.cat([v_packed_q, v_res_q], dim=2)
        v_s_all = torch.cat([v_scale_q, v_res_s], dim=2)
        return k_q_all, k_s_all, v_q_all, v_s_all

    def stored_bytes(self, layer_idx: int) -> int:
        """Total bytes stored for `layer_idx` (for memory reporting). Counts
        packed int4 K/V (1/2 byte/elem), scale tensors (2 bytes/elem fp16),
        and fp16 residual (2 bytes/elem). Useful for the RESULTS.md memory row."""
        total = 0
        for t in (self.k_packed[layer_idx], self.v_packed[layer_idx]):
            if t is not None:
                total += t.numel()  # 1 byte/elem (int8 container, holds 2 int4)
        for t in (self.k_scale[layer_idx], self.v_scale[layer_idx]):
            if t is not None:
                total += t.numel() * 2  # fp16
        for t in (self.k_resid[layer_idx], self.v_resid[layer_idx]):
            if t is not None:
                total += t.numel() * 2  # fp16
        return total

    # ---- internal helpers ----

    def _materialize_fp16(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize the int4 portion + concat the fp16 residual. Returns
        (K_full, V_full) fp16, both [B, n_kv_heads, seq_total, head_dim].
        Used by `update()` for SDPA path compat (prefill)."""
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []
        if self.k_packed[layer_idx] is not None:
            k_unpacked = _unpack_int4_packed(self.k_packed[layer_idx])
            k_dequant = _dequantize_per_channel_groupwise(
                k_unpacked, self.k_scale[layer_idx], self.group_size)
            v_unpacked = _unpack_int4_packed(self.v_packed[layer_idx])
            v_dequant = _dequantize_per_token(v_unpacked, self.v_scale[layer_idx])
            parts_k.append(k_dequant)
            parts_v.append(v_dequant)
        if self.k_resid[layer_idx] is not None and self.k_resid[layer_idx].shape[2] > 0:
            parts_k.append(self.k_resid[layer_idx])
            parts_v.append(self.v_resid[layer_idx])
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)
