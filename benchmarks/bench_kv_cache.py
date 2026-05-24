"""Benchmark: INT8 KV-cache vs fp16 KV (decode attention).

Measures:
  - PyTorch SDPA on fp16 KV  (production baseline)
  - v3 fp16 attention        (Phase 1 main)
  - INT8 attention + fused dequant
  - Quantize kernel one-shot cost

For each, reports median latency and the steady-state KV-cache size on the
GPU (the primary metric for Phase 2 — Phase 2's win is memory, not latency).
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from harness import benchmark, check_close, achieved_bandwidth_gbps  # noqa: E402
from reference.attention_ref import decode_attention, sdpa_attention  # noqa: E402
from reference.kv_cache_ref import quantize_per_token, dequantize_per_token  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA device required.")
        return

    import llmik_cuda  # local import so the script imports cleanly without the build

    torch.manual_seed(0)
    # Llama 3 8B head config; same reference workload as bench_attention.py.
    batch, n_heads, n_kv_heads, head_dim = 8, 32, 8, 128
    seqlen_kv = 4096
    dev   = "cuda"
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, n_heads,    head_dim,             device=dev, dtype=torch.float16)
    k = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,  device=dev, dtype=torch.float16)
    v = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,  device=dev, dtype=torch.float16)

    # Pre-quantize K and V once: in serving you'd quantize on-append (one
    # new token per decode step), but that one-shot cost is reported below.
    k_q, k_s = llmik_cuda.quantize_per_token(k)
    v_q, v_s = llmik_cuda.quantize_per_token(v)

    fp16_kv_bytes  = (k.numel() + v.numel()) * 2
    int8_kv_bytes  = (k_q.numel() + v_q.numel()) + (k_s.numel() + v_s.numel()) * 2

    print(f"workload: batch={batch} n_heads={n_heads} kv_heads={n_kv_heads} "
          f"head_dim={head_dim} seqlen_kv={seqlen_kv}\n")
    print(f"  KV size  fp16: {fp16_kv_bytes / 1024 / 1024:6.2f} MiB")
    print(f"  KV size  int8: {int8_kv_bytes / 1024 / 1024:6.2f} MiB "
          f"({int8_kv_bytes / fp16_kv_bytes:.2f}× of fp16, "
          f"{(fp16_kv_bytes - int8_kv_bytes) / 1024 / 1024:.2f} MiB saved)\n")

    # --- correctness sanity ---
    ref = decode_attention(q.unsqueeze(2), k, v, scale=scale).squeeze(2)
    out_fp16 = llmik_cuda.decode_attention(q, k, v, scale)
    check_close(out_fp16, ref, name="v3 fp16 KV")
    # INT8: compare against the dequantize-then-attention reference (same
    # dequantized values as the kernel sees, so this isolates kernel
    # correctness from quantization noise).
    k_dq = (k_q.float() * k_s.float().unsqueeze(-1)).to(torch.float16)
    v_dq = (v_q.float() * v_s.float().unsqueeze(-1)).to(torch.float16)
    ref_int8 = decode_attention(q.unsqueeze(2), k_dq, v_dq, scale=scale).squeeze(2)
    out_int8 = llmik_cuda.decode_attention_int8(q, k_q, k_s, v_q, v_s, scale)
    check_close(out_int8, ref_int8, name="INT8 KV + fused dequant")

    # Also measure the end-to-end accuracy delta (INT8 quantize → attention)
    # vs the fp16 reference. This is the "what does INT8 cost in answer
    # quality" number at the kernel level.
    abs_err = (out_int8.float() - ref.float()).abs()
    rel_err = (abs_err / ref.float().abs().clamp(min=1e-3)).mean().item()
    print(f"\n  INT8 vs fp16 reference: max |diff| {abs_err.max().item():.3e}, "
          f"mean rel err {rel_err:.3e}")
    print(f"  Per-token quantization noise: ~1/127 of per-token max ≈ {1/127:.4f}")

    # --- latency ---
    print()
    print(benchmark(lambda: sdpa_attention(q.unsqueeze(2), k, v),
                    name="pytorch sdpa (fp16)"))
    print(benchmark(lambda: llmik_cuda.decode_attention(q, k, v, scale),
                    name="v3 fp16 KV"))
    res_int8 = benchmark(
        lambda: llmik_cuda.decode_attention_int8(q, k_q, k_s, v_q, v_s, scale),
        name="INT8 KV + fused dequant")
    print(res_int8)
    print(f"  INT8 achieved KV bandwidth: "
          f"{achieved_bandwidth_gbps(int8_kv_bytes, res_int8.median_ms):.0f} GB/s")

    # Quantize cost (one-shot; in serving this runs per appended token).
    print(benchmark(lambda: llmik_cuda.quantize_per_token(k),
                    name="quantize_per_token (K only)"))


if __name__ == "__main__":
    main()
