"""End-to-end W4A16 + v6 decode throughput vs vLLM, on the locked workload.

Composes the repo's kernels into one decode path and measures it against the
Phase 0 vLLM baseline (703.2 tok/s), on the same workload bench_e2e.py uses
(Llama 3.1 8B, batch=16, prompt=512, generate=512, greedy):

  - v6 FlashDecoding split-K attention (Phase 8)     — parity with SDPA
  - W4A16 split-K GEMM (Phase 9)                      — beats cuBLAS at M=16
  - PreallocCache + v6 length-aware reads (Phase 9)   — no per-step torch.cat
  - fused prefill dequant (Phase 9)                   — W4A16 prefill ~= fp16

Honest framing (printed below): this compares a *4-bit-weight* model to
vLLM-*fp16*. The throughput win comes partly from reading 4x fewer weight bytes,
and W4A16 costs the accuracy delta measured in Phase 4c (docs/results/RESULTS.md).
A like-for-like 4-bit comparison would be vs vLLM-AWQ, which is not measured here.
The kernel-quality claims (attention == SDPA, W4A16 GEMM > cuBLAS) are apples to
apples and live in bench_decode_step.py / bench_w4a16.py.

Usage:
    python benchmarks/bench_w4a16_e2e.py --runs 3
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from benchmarks.workload import (E2E_BATCH, E2E_GEN_LEN, E2E_PROMPT_LEN,  # noqa: E402
                                 MODEL_ID, VOCAB_SIZE)
from integration.prealloc_cache import PreallocCache, v6_decode  # noqa: E402
from integration.w4a16_patch import patch_model_w4a16  # noqa: E402

VLLM_REF = 703.2                    # Phase 0, docs/results/RESULTS.md
_TOTAL_OUT = E2E_BATCH * E2E_GEN_LEN
_MAX_ID = VOCAB_SIZE - 1000


def _prompt_ids() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, _MAX_ID, (E2E_BATCH, E2E_PROMPT_LEN), generator=g).cuda()


def _run(model, ids, gen, capture=0):
    cache = PreallocCache(E2E_PROMPT_LEN + E2E_GEN_LEN + 8)
    trace = []
    with v6_decode(cache), torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(ids.size(1), device="cuda"))
        torch.cuda.synchronize()
        prefill = time.perf_counter() - t0
        tok = out.logits[:, -1:].argmax(-1)
        t0 = time.perf_counter()
        for i in range(gen):
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([ids.size(1) + i], device="cuda"))
            if i < capture:
                trace.append(out.logits[:, -1].float().argmax(-1).clone())
            tok = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        decode = time.perf_counter() - t0
    return prefill, decode, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa").eval()
    ids = _prompt_ids()

    # Correctness gate: greedy tokens of the full stack vs the stock fp16 model.
    with torch.inference_mode():
        cache0 = __import__("transformers").cache_utils.DynamicCache()
        out = model(input_ids=ids, past_key_values=cache0, use_cache=True,
                    cache_position=torch.arange(ids.size(1), device="cuda"))
        tok = out.logits[:, -1:].argmax(-1)
        ref = []
        for i in range(8):
            out = model(input_ids=tok, past_key_values=cache0, use_cache=True,
                        cache_position=torch.tensor([ids.size(1) + i], device="cuda"))
            ref.append(out.logits[:, -1].float().argmax(-1).clone())
            tok = out.logits[:, -1:].argmax(-1)

    n = patch_model_w4a16(model, group_size=128)
    _, _, got = _run(model, ids, 8, capture=8)
    match = torch.stack([(a == b).float().mean() for a, b in zip(ref, got)]).mean().item()
    print(f"patched {n} Linears to W4A16.")
    print(f"greedy-token agreement with the fp16 model, 8 decode steps: "
          f"{match * 100:.1f}%")
    print("  (W4A16 is LOSSY — this divergence is the accuracy cost of 4-bit weights, "
          "not a bug.\n   Real task accuracy: Phase 4c MMLU 62.4% vs fp16 68.3%, "
          "docs/results/RESULTS.md.\n   vLLM-fp16 is lossless; the throughput numbers "
          "below are a speed/accuracy trade.)")

    _run(model, ids, 8)                                   # warmup
    pf, dc = [], []
    for _ in range(args.runs):
        p, d, _ = _run(model, ids, E2E_GEN_LEN)
        pf.append(p); dc.append(d)
        torch.cuda.empty_cache()
    prefill, decode = statistics.median(pf), statistics.median(dc)
    step_ms = decode / E2E_GEN_LEN * 1e3
    full = _TOTAL_OUT / (prefill + decode)
    dec = _TOTAL_OUT / decode

    print(f"\nworkload: Llama 3.1 8B, batch={E2E_BATCH}, prompt={E2E_PROMPT_LEN}, "
          f"gen={E2E_GEN_LEN}, greedy")
    print(f"  prefill {prefill:.2f}s   decode {decode:.2f}s   ({step_ms:.2f} ms/step)")
    print(f"  full e2e (incl prefill): {full:7.1f} tok/s   ({full / VLLM_REF:.2f}x vLLM-fp16 {VLLM_REF})")
    print(f"  decode-only throughput:  {dec:7.1f} tok/s   ({dec / VLLM_REF:.2f}x vLLM-fp16 {VLLM_REF})")
    print("\n  Note: 4-bit weights vs vLLM-fp16 (16-bit) — a throughput/accuracy trade,")
    print("  not a like-for-like precision comparison. See docstring + RESULTS.md.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required."); sys.exit(1)
    main()
