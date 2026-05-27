"""Phase 4c: W4A16 weight quantization for Llama 3 linear projections.

Replaces each `nn.Linear` inside `LlamaDecoderLayer` (q/k/v/o_proj on attention,
up/gate/down_proj on MLP) with a `QuantizedLinear`:

  - Stores packed INT4 weights + per-channel groupwise scales (group=128 along K)
    — the Phase 3a/b/c scheme.
  - Forward dispatches:
      M == 1 (decode)  → `llmik_cuda.w4a16_gemm` (Phase 3c decode-optimized)
      M  > 1 (prefill) → dequantize-on-the-fly → cuBLAS fp16 GEMM

Embeddings + `lm_head` are NOT quantized — they're large but only run once per
token (no decode dominance), and quantizing lm_head would also break the
shared-weight tie if the model uses it.

Memory: full Llama 3.1 8B Instruct fp16 weight is ~16 GB. After 4c, the
per-layer Linears land at ~0.27× their original size (int4 + scales). Embed
+ lm_head stay fp16. Net ≈ 6-7 GB savings on weight storage.

Usage:
    from integration.w4a16_patch import patch_model_w4a16, restore_model_w4a16

    handles = patch_model_w4a16(model, group_size=128)
    # ... run eval ...
    restore_model_w4a16(model, handles)
"""
from __future__ import annotations

import sys
import os
from typing import List, Tuple

import torch
import torch.nn as nn

import llmik_cuda

# Reference quantize/pack helpers (pure Python; one-time at patch time)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from reference.quant_matmul_ref import (  # noqa: E402
    pack_int4_along_k,
    quantize_weights_int4_groupwise,
)

# Linear layers within a LlamaDecoderLayer that we quantize. Names match
# `LlamaAttention` and `LlamaMLP` attribute names. All have bias=False in
# Llama 3 (verified before commit).
_TARGET_LINEAR_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",       # LlamaAttention
    "up_proj", "gate_proj", "down_proj",          # LlamaMLP
}


class QuantizedLinear(nn.Module):
    """W4A16 drop-in replacement for `nn.Linear(bias=False)`.

    Stores `[K/8, N] int32` packed weights + `[n_groups, N] fp16` scales.
    Forward routes to our `w4a16_gemm` for decode (M==1); for prefill (M>1)
    it dequantizes the weight once into a transient fp16 buffer and uses
    `torch.matmul` (cuBLAS).
    """

    def __init__(self, weight_fp16: torch.Tensor, group_size: int = 128) -> None:
        super().__init__()
        # `weight_fp16` is in `nn.Linear` layout: [N=out_features, K=in_features].
        N, K = weight_fp16.shape
        assert K % group_size == 0, (
            f"in_features={K} must be a multiple of group_size={group_size}")
        assert K % 8 == 0, f"K={K} must be a multiple of 8 to pack int4"

        self.in_features = K
        self.out_features = N
        self.group_size = group_size

        # Transpose to the reference's [K, N] layout, quantize, pack
        W_kn = weight_fp16.detach().T.contiguous().float()
        q_int8, scale = quantize_weights_int4_groupwise(W_kn, group_size=group_size)
        packed = pack_int4_along_k(q_int8.to(torch.int8))

        # Register as buffers (not parameters — frozen weights)
        self.register_buffer("weight_packed", packed.contiguous())          # [K/8, N] int32
        self.register_buffer("scale", scale.contiguous().to(torch.float16))  # [n_groups, N] fp16

    # Above this M, the naive kernel becomes slower than a one-shot dequant +
    # cuBLAS. Empirically tuned: e2e prefill (M=batch*prompt_len = 16*512 = 8192)
    # needs the dequant path; per-token decode (M=batch=16) wants the kernel.
    _DEQUANT_PATH_M_THRESHOLD = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., K]
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features).contiguous()
        M = x_flat.shape[0]
        if M < self._DEQUANT_PATH_M_THRESHOLD:
            # Decode-ish path: M==1 hits the decode-optimized kernel; the
            # launcher falls back to the naive kernel for M>1 (still beats the
            # PyTorch-side dequant for small M).
            out = llmik_cuda.w4a16_gemm(
                x_flat, self.weight_packed, self.scale, self.group_size)
        else:
            # Prefill: dequantize on-the-fly, cuBLAS GEMM. The naive kernel
            # degrades linearly with M, while one dequant + cuBLAS is roughly
            # constant — crossover is around M~256 on this GPU.
            W_dequant = self._dequantize_to_fp16()
            out = torch.matmul(x_flat, W_dequant)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _dequantize_to_fp16(self) -> torch.Tensor:
        """Unpack `[K/8, N] int32` → fp16 `[K, N]`, multiplied by per-group scale.
        Allocated fresh each call; freed after the GEMM in forward()."""
        K_packs, N = self.weight_packed.shape
        K = K_packs * 8
        # Unpack: 8 nibbles per int32. Bit-shift trick mirrors the CUDA kernel's
        # unpack inner loop, vectorized in PyTorch.
        packed = self.weight_packed.unsqueeze(1)  # [K/8, 1, N]
        shifts = torch.arange(8, device=packed.device).view(1, 8, 1) * 4
        nibbles = (packed >> shifts) & 0xF  # [K/8, 8, N], unsigned
        # Sign-extend 4-bit values: [-7, 7] (8 means -8 which we don't produce
        # since qmax=7, but handle for safety)
        signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).to(torch.int8)
        q_int8 = signed.reshape(K, N).contiguous()

        # Per-group scale: K positions in group g share scale[g, :]
        group_idx = torch.arange(K, device=packed.device) // self.group_size
        scale_expanded = self.scale.index_select(dim=0, index=group_idx)
        W = q_int8.to(torch.float16) * scale_expanded
        return W.to(torch.float16)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"group_size={self.group_size}, bits=4")


def patch_model_w4a16(model: nn.Module, *, group_size: int = 128) -> int:
    """Walk `model.model.layers.*.{self_attn,mlp}.*_proj` and replace each
    `nn.Linear` with a `QuantizedLinear`. The original Linear modules are
    NOT retained (so their fp16 weights can be garbage-collected and the
    memory savings actually materialize). To "restore" you must reload the
    model from disk — this patch is one-way by design.

    Args:
        model: HF LlamaForCausalLM (or similar with `model.model.layers`).
        group_size: K-axis group size for INT4 quant; matches the kernel
            assumption (typically 128).

    Returns:
        Number of Linear modules replaced.
    """
    replaced = 0
    for layer in model.model.layers:
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name)
            for child_name, child in list(parent.named_children()):
                if child_name not in _TARGET_LINEAR_NAMES:
                    continue
                if not isinstance(child, nn.Linear):
                    continue
                assert child.bias is None, (
                    f"unexpected bias on {parent_name}.{child_name}; "
                    "QuantizedLinear assumes bias=False")
                qlin = QuantizedLinear(child.weight, group_size=group_size).to(
                    device=child.weight.device)
                setattr(parent, child_name, qlin)
                # Drop the local reference; `setattr` already replaced the
                # parent's hold on `child`. With no strong refs left, the
                # fp16 weight tensor frees on the next GC pass / empty_cache.
                del child
                replaced += 1
    torch.cuda.empty_cache()
    return replaced


def quantized_weight_bytes(model: nn.Module) -> int:
    """Sum the bytes of all QuantizedLinear buffers in `model`. Used for
    reporting the W4A16 memory savings."""
    total = 0
    for mod in model.modules():
        if isinstance(mod, QuantizedLinear):
            total += mod.weight_packed.numel() * 4  # int32 = 4 bytes
            total += mod.scale.numel() * 2          # fp16 = 2 bytes
    return total
