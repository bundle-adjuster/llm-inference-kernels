"""Benchmark: fused decode attention vs PyTorch baselines.

Phase 0: runs the PyTorch eager baseline so the harness is exercised.
Phase 1: uncomment the custom-kernel block once the extension is built.
Copy the numbers into docs/results/RESULTS.md.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from harness import benchmark, check_close, achieved_bandwidth_gbps  # noqa: E402
from reference.attention_ref import decode_attention, sdpa_attention  # noqa: E402


def main() -> None:
    """Bench whichever decode-attention kernel is currently built.

    Compares the custom kernel against PyTorch eager and PyTorch SDPA on
    the locked microbench workload (Llama 3 8B head config, `batch=8,
    seqlen_kv=4096`). Prints median latency + achieved KV bandwidth.
    """
    if not torch.cuda.is_available():
        print("CUDA device required.")
        return

    torch.manual_seed(0)
    batch, n_heads, n_kv_heads, head_dim = 8, 32, 8, 128  # Llama 3 8B config
    seqlen_kv = 4096
    dev = "cuda"
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, n_heads, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                    device=dev, dtype=torch.float16)
    v = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                    device=dev, dtype=torch.float16)

    print(f"decode attention | batch={batch} heads={n_heads} "
          f"kv_heads={n_kv_heads} head_dim={head_dim} seqlen_kv={seqlen_kv}")

    # --- baseline: PyTorch eager reference ---
    print(benchmark(lambda: decode_attention(q.unsqueeze(2), k, v, scale=scale),
                    name="pytorch eager (decode)"))

    # --- baseline: PyTorch SDPA (FlashAttention / cuDNN) ---
    # NOTE: `sdpa_attention` first calls `_expand_gqa`, so SDPA runs on a
    # 4x-expanded KV cache. That is the HANDICAPPED baseline (Phase 7): the
    # old v3 kernel only looked "1.91x faster than SDPA" because SDPA was
    # forced to read 4x the KV bytes. Against GQA-native SDPA
    # (F.scaled_dot_product_attention(..., enable_gqa=True)) v3 actually LOST
    # (0.22x). The v6 split-K kernel is what finally beats fair SDPA (1.01x).
    # For the fair GQA-native comparison, run benchmarks/bench_decode_step.py.
    # See docs/05-baseline-correction-journey.md and
    # docs/06-attention-splitk-journey.md.
    print(benchmark(lambda: sdpa_attention(q.unsqueeze(2), k, v),
                    name="pytorch sdpa (decode, GQA-EXPANDED / handicapped)"))

    # --- custom kernel (v0) ---
    import llmik_cuda
    ref = decode_attention(q.unsqueeze(2), k, v, scale=scale).squeeze(2)
    out = llmik_cuda.decode_attention(q, k, v, scale)
    check_close(out, ref, name="custom decode kernel")
    res = benchmark(lambda: llmik_cuda.decode_attention(q, k, v, scale),
                    name="custom decode kernel")
    print(res)
    kv_bytes = 2 * k.numel() * k.element_size()  # K + V read once
    print(f"  achieved KV bandwidth: "
          f"{achieved_bandwidth_gbps(kv_bytes, res.median_ms):.0f} GB/s")


if __name__ == "__main__":
    main()
