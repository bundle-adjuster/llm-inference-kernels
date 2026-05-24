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
    """Bench fp16 / INT8 / INT4 KIVI KV-cache paths side by side.

    Reports KV-cache memory, end-to-end attention latency, achieved KV
    bandwidth, and the per-token quantize one-shot cost for each path.
    Includes a kernel-level accuracy check (CUDA vs reference attention
    on the same dequantized inputs) before timing.
    """
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
    k_q8,  k_s8  = llmik_cuda.quantize_per_token(k)
    v_q8,  v_s8  = llmik_cuda.quantize_per_token(v)
    int4_group_size = 32
    k_q4,  k_s4  = llmik_cuda.quantize_k_per_channel_groupwise_int4(k, int4_group_size)
    v_q4,  v_s4  = llmik_cuda.quantize_v_per_token_int4(v)

    fp16_kv_bytes = (k.numel() + v.numel()) * 2
    int8_kv_bytes = (k_q8.numel() + v_q8.numel()) + (k_s8.numel() + v_s8.numel()) * 2
    int4_kv_bytes = (k_q4.numel() + v_q4.numel()) + (k_s4.numel() + v_s4.numel()) * 2

    print(f"workload: batch={batch} n_heads={n_heads} kv_heads={n_kv_heads} "
          f"head_dim={head_dim} seqlen_kv={seqlen_kv}\n")
    print(f"  KV size  fp16: {fp16_kv_bytes / 1024 / 1024:6.2f} MiB")
    print(f"  KV size  int8: {int8_kv_bytes / 1024 / 1024:6.2f} MiB "
          f"({int8_kv_bytes / fp16_kv_bytes:.2f}× of fp16, "
          f"{(fp16_kv_bytes - int8_kv_bytes) / 1024 / 1024:.2f} MiB saved)")
    print(f"  KV size  int4: {int4_kv_bytes / 1024 / 1024:6.2f} MiB "
          f"({int4_kv_bytes / fp16_kv_bytes:.2f}× of fp16, "
          f"{(fp16_kv_bytes - int4_kv_bytes) / 1024 / 1024:.2f} MiB saved)"
          f"   [int4 group_size={int4_group_size}]\n")

    # --- correctness sanity ---
    ref = decode_attention(q.unsqueeze(2), k, v, scale=scale).squeeze(2)
    out_fp16 = llmik_cuda.decode_attention(q, k, v, scale)
    check_close(out_fp16, ref, name="v3 fp16 KV")

    # INT8: compare against fp16 attention on the dequantized values (same
    # dequant the kernel sees, isolates kernel correctness from quant noise).
    k_dq8 = (k_q8.float() * k_s8.float().unsqueeze(-1)).to(torch.float16)
    v_dq8 = (v_q8.float() * v_s8.float().unsqueeze(-1)).to(torch.float16)
    ref_int8 = decode_attention(q.unsqueeze(2), k_dq8, v_dq8, scale=scale).squeeze(2)
    out_int8 = llmik_cuda.decode_attention_int8(q, k_q8, k_s8, v_q8, v_s8, scale)
    check_close(out_int8, ref_int8, name="INT8 KV + fused dequant",
                rtol=2e-2, atol=2e-2)

    # INT4 KIVI accuracy delta vs fp16 reference (the "what does INT4 cost"
    # number at the kernel level).
    out_int4 = llmik_cuda.decode_attention_int4(q, k_q4, k_s4, v_q4, v_s4,
                                                int4_group_size, scale)
    abs_err8 = (out_int8.float() - ref.float()).abs()
    rel_err8 = (abs_err8 / ref.float().abs().clamp(min=1e-3)).mean().item()
    abs_err4 = (out_int4.float() - ref.float()).abs()
    rel_err4 = (abs_err4 / ref.float().abs().clamp(min=1e-3)).mean().item()
    print(f"\n  INT8 vs fp16 reference: max |diff| {abs_err8.max().item():.3e}, "
          f"mean rel err {rel_err8:.3e}")
    print(f"  INT4 vs fp16 reference: max |diff| {abs_err4.max().item():.3e}, "
          f"mean rel err {rel_err4:.3e}")
    print(f"  Quantization noise floor: INT8 ≈ 1/127 = {1/127:.4f}, "
          f"INT4 ≈ 1/7 = {1/7:.4f}")

    # --- latency ---
    print()
    print(benchmark(lambda: sdpa_attention(q.unsqueeze(2), k, v),
                    name="pytorch sdpa (fp16)"))
    print(benchmark(lambda: llmik_cuda.decode_attention(q, k, v, scale),
                    name="v3 fp16 KV"))
    res_int8 = benchmark(
        lambda: llmik_cuda.decode_attention_int8(q, k_q8, k_s8, v_q8, v_s8, scale),
        name="INT8 KV + fused dequant")
    print(res_int8)
    print(f"  INT8 achieved KV bandwidth: "
          f"{achieved_bandwidth_gbps(int8_kv_bytes, res_int8.median_ms):.0f} GB/s")
    res_int4 = benchmark(
        lambda: llmik_cuda.decode_attention_int4(q, k_q4, k_s4, v_q4, v_s4,
                                                 int4_group_size, scale),
        name="INT4 KV (KIVI) + fused dequant")
    print(res_int4)
    print(f"  INT4 achieved KV bandwidth: "
          f"{achieved_bandwidth_gbps(int4_kv_bytes, res_int4.median_ms):.0f} GB/s")

    # Quantize one-shot costs (in serving these amortise to per-appended-token).
    print()
    print(benchmark(lambda: llmik_cuda.quantize_per_token(k),
                    name="quantize_per_token int8 (K)"))
    print(benchmark(
        lambda: llmik_cuda.quantize_k_per_channel_groupwise_int4(k, int4_group_size),
        name="quantize_k int4 (per-channel)"))
    print(benchmark(lambda: llmik_cuda.quantize_v_per_token_int4(v),
                    name="quantize_v int4 (per-token)"))


if __name__ == "__main__":
    main()
