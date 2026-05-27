"""Phase 2d: WikiText-2 perplexity for Llama 3.1 8B with KV-cache quantization.

Evaluates how each KV-cache compression mode affects perplexity on WikiText-2:

  fp16            — baseline (no quantization)
  int8            — per-token symmetric quantize+dequant of K and V
  int4-per-token  — same shape as int8 but qmax=7 (worst-case INT4 baseline)
  int4-kivi       — K per-channel groupwise (group_size=32) + V per-token,
                    bits=4. This is the headline of Phase 2c.

Method: prefill-only forward passes. For each 2048-token chunk of the
WikiText-2 test split, compute the average cross-entropy loss the model
would assign to the chunk if it always *read* a quantized KV cache (the
attention sees noisy K, V — exactly what would come out of dequantizing
a compressed cache at decode time).

Implementation:
  - The model is loaded with attn_implementation="sdpa" so all attention
    flows through `F.scaled_dot_product_attention`.
  - We rebind `torch.nn.functional.scaled_dot_product_attention` to a
    wrapper that quantizes+dequantizes K and V before delegating to the
    original sdpa. LlamaSdpaAttention.forward looks up the symbol on the
    F module at call time, so the rebind intercepts every layer's attention.
  - The quantization uses the PyTorch reference in `reference/kv_cache_ref.py`
    — same math as the CUDA kernels (we proved that in Phase 2b/2c tests),
    so the perplexity number is faithful to what running with the CUDA
    INT8/INT4 KV path would produce.

Usage:
    python scripts/eval_perplexity.py [--n-chunks N] [--chunk-size N]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Callable, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.workload import MODEL_ID
from reference.kv_cache_ref import (
    dequantize_per_channel_groupwise,
    dequantize_per_token,
    quantize_per_channel_groupwise,
    quantize_per_token,
)

_ORIGINAL_SDPA = F.scaled_dot_product_attention


def _quantize_and_back(x: torch.Tensor, mode: str, *,
                       bits: int, group_size: int) -> torch.Tensor:
    """Round-trip quantize → dequantize a K or V tensor.

    Simulates "the cache was compressed in `mode`": the returned tensor
    has the same shape and dtype as `x`, but its values are noisy in the
    way the model would see them if the KV cache stored quantized bytes
    and dequantized on the read side.

    Args:
        x: `[batch, n_heads, seqlen, head_dim]` (or any shape ending in
            `head_dim`) — works for K and V identically.
        mode: "per_token" (one scale per (batch, head, token), shared
            across head_dim) or "per_channel_groupwise" (one scale per
            (batch, head, group_of_tokens, head_dim_channel) — KIVI's K
            recipe).
        bits: 4 or 8.
        group_size: tokens per group; only used in per_channel_groupwise.

    Returns:
        Same shape and dtype as `x`, values noisy per the chosen scheme.
    """
    dtype = x.dtype
    if mode == "per_token":
        q, s = quantize_per_token(x.float(), bits=bits)
        return dequantize_per_token(q, s.float()).to(dtype)
    elif mode == "per_channel_groupwise":
        q, s = quantize_per_channel_groupwise(x.float(), bits=bits,
                                              group_size=group_size)
        return dequantize_per_channel_groupwise(q, s.float(),
                                                group_size).to(dtype)
    else:
        raise ValueError(f"unknown mode {mode}")


SdpaFn = Callable[..., torch.Tensor]


def make_patched_sdpa(mode: str, *, group_size: int = 32) -> SdpaFn:
    """Return a callable with the F.scaled_dot_product_attention signature
    that simulates `mode`'s KV-cache compression by quantizing+dequantizing
    K and V before the underlying attention.

    Args:
        mode: one of "fp16" (no patch — returns the original sdpa),
            "int8", "int4-per-token", "int4-kivi".
        group_size: tokens per group for the KIVI per-channel K path.

    Returns:
        A callable to bind to `F.scaled_dot_product_attention` via `_set_sdpa`.
        Falls back to `_ORIGINAL_SDPA` for `mode == "fp16"`.
    """
    if mode == "fp16":
        return _ORIGINAL_SDPA

    def patched(query, key, value, *args, **kwargs):
        if mode == "int8":
            key   = _quantize_and_back(key,   "per_token", bits=8, group_size=group_size)
            value = _quantize_and_back(value, "per_token", bits=8, group_size=group_size)
        elif mode == "int4-per-token":
            key   = _quantize_and_back(key,   "per_token", bits=4, group_size=group_size)
            value = _quantize_and_back(value, "per_token", bits=4, group_size=group_size)
        elif mode == "int4-kivi":
            # KIVI: K per-channel groupwise, V per-token.
            key   = _quantize_and_back(key,   "per_channel_groupwise",
                                       bits=4, group_size=group_size)
            value = _quantize_and_back(value, "per_token", bits=4, group_size=group_size)
        else:
            raise ValueError(f"unknown mode {mode}")
        return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

    return patched


def _set_sdpa(fn: SdpaFn) -> None:
    """Rebind F.scaled_dot_product_attention everywhere it's looked up at
    call time. Both `F.…` and `torch.nn.functional.…` point at the same
    module object, so assigning to either is sufficient — we do both for
    paranoia (some old code paths may have cached the symbol)."""
    F.scaled_dot_product_attention = fn
    torch.nn.functional.scaled_dot_product_attention = fn


def evaluate_perplexity(
    model: torch.nn.Module,
    chunks: List[torch.Tensor],
    label: str,
    cache_factory: Callable[[], "object"] | None = None,
) -> float:
    """Forward each chunk once, sum the cross-entropy loss, return perplexity.

    Args:
        model: causal LM in eval mode on CUDA.
        chunks: list of 1-D `[chunk_size]` int64 token tensors on CPU; each
            chunk is `.unsqueeze(0).cuda()`d into a `[1, chunk_size]` batch.
        label: printed alongside the result; should describe the patch state.
        cache_factory: optional zero-arg callable returning a fresh HF Cache;
            used by Phase 4b to inject Int4KIVICache so the forward exercises
            the quantized cache path. If None, the model uses its default
            (no cache for a single-chunk forward).

    Returns:
        Average perplexity across all chunks, weighted by token count.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    for chunk in chunks:
        input_ids = chunk.unsqueeze(0).cuda()
        kwargs = {"labels": input_ids}
        if cache_factory is not None:
            kwargs["past_key_values"] = cache_factory()
            kwargs["use_cache"] = True
        with torch.no_grad():
            out = model(input_ids, **kwargs)
        n_tokens = input_ids.size(1) - 1
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    elapsed = time.time() - t0
    ppl = math.exp(total_loss / total_tokens)
    print(f"  {label:>16}:  ppl = {ppl:7.3f}   ({total_tokens} tokens, "
          f"{elapsed:5.1f} s, {total_tokens/elapsed:.0f} tok/s)")
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-chunks", type=int, default=64,
                        help="number of 2048-token chunks of WikiText-2 to eval")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="context length per eval chunk")
    parser.add_argument("--group-size", type=int, default=32,
                        help="KIVI group size for per-channel K quantization")
    parser.add_argument("--modes", nargs="+",
                        default=["fp16", "int8", "int4-per-token", "int4-kivi"],
                        help="which compression modes to evaluate")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    print(f"loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded; vram peak so far: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"loading WikiText-2 test split ...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in (x["text"] for x in raw) if t.strip())
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

    n_possible = len(full_ids) // args.chunk_size
    n_chunks = min(args.n_chunks, n_possible)
    chunks = [full_ids[i * args.chunk_size:(i + 1) * args.chunk_size]
              for i in range(n_chunks)]
    n_tokens_total = n_chunks * args.chunk_size
    print(f"  using {n_chunks} chunks of {args.chunk_size} tokens "
          f"({n_tokens_total} tokens, {len(full_ids)} available)")
    print()

    results: dict[str, float] = {}
    for mode in args.modes:
        _set_sdpa(make_patched_sdpa(mode, group_size=args.group_size))
        try:
            ppl = evaluate_perplexity(model, chunks, label=mode)
        finally:
            _set_sdpa(_ORIGINAL_SDPA)
        results[mode] = ppl

    if "fp16" in results:
        base = results["fp16"]
        print()
        print(f"  baseline (fp16): {base:.3f}")
        for mode, ppl in results.items():
            if mode == "fp16":
                continue
            delta = ppl - base
            print(f"  {mode:>16}: Δppl = {delta:+.3f} ({100 * delta / base:+.2f}%)")


if __name__ == "__main__":
    main()
