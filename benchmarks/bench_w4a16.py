"""Benchmark: W4A16 quantized matmul vs FP16 cuBLAS.

Measures the W4A16 GEMM (this repo) against `torch.matmul` (cuBLAS fp16
under the hood) across:
  - Llama 3 8B linear-layer shapes: attn (4096×4096), MLP up/gate
    (4096×14336), MLP down (14336×4096)
  - M sweep: 1 (decode), 8, 32 (small batched / prefill probe)

For each (shape, M), reports:
  - fp16 baseline latency
  - W4A16 kernel latency
  - speedup vs fp16
  - achieved weight bandwidth (interpretable for memory-bound regime)

Phase 3 has *two* paths in the W4A16 launcher:
  - M == 1: the decode-optimized kernel (Phase 3c). 4-warp blocks, K
    split across warps, act cached in shmem.
  - M > 1: the naive Phase 3b kernel. Iterates M sequentially per output
    column — slow on purpose; M > 1 is a Phase 4 integration concern.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from harness import benchmark, achieved_bandwidth_gbps  # noqa: E402
from reference.quant_matmul_ref import (  # noqa: E402
    pack_int4_along_k,
    quantize_weights_int4_groupwise,
)


# Llama 3 8B linear-layer shapes the kernel targets.
SHAPES = [
    ("attn-qkv-or-out", 4096,  4096),
    ("mlp-up-or-gate",  4096, 14336),
    ("mlp-down",       14336,  4096),
]
M_SWEEP = [1, 8, 32]
GROUP_SIZE = 128


def main() -> None:
    """Run the W4A16 vs cuBLAS sweep and print a side-by-side table.

    The output is the canonical Phase 3 measurement; copy the M=1 rows
    into the Track 3 table in `docs/results/RESULTS.md`.
    """
    if not torch.cuda.is_available():
        print("CUDA device required.")
        return

    import llmik_cuda  # local import so the script imports without the build

    torch.manual_seed(0)

    for name, K, N in SHAPES:
        # Quantize weights once per shape.
        W = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
        q, scale = quantize_weights_int4_groupwise(W, group_size=GROUP_SIZE)
        packed = pack_int4_along_k(q)

        fp16_w_bytes  = W.numel() * 2
        packed_bytes  = packed.numel() * 4 + scale.numel() * 2
        weight_ratio  = packed_bytes / fp16_w_bytes

        print(f"\n=== {name}: K={K}, N={N} ===")
        print(f"  fp16 W:     {fp16_w_bytes  / 1024 / 1024:7.2f} MiB")
        print(f"  W4A16 W:    {packed_bytes  / 1024 / 1024:7.2f} MiB "
              f"({weight_ratio:.3f}× fp16, "
              f"{(fp16_w_bytes - packed_bytes) / 1024 / 1024:.1f} MiB saved)")

        for M in M_SWEEP:
            act = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
            r_fp16 = benchmark(lambda: act @ W,
                               name=f"  M={M:>2}  fp16 cuBLAS")
            r_w4a16 = benchmark(
                lambda: llmik_cuda.w4a16_gemm(act, packed, scale, GROUP_SIZE),
                name=f"  M={M:>2}  w4a16")
            speedup = r_fp16.median_ms / r_w4a16.median_ms
            print(r_fp16)
            print(r_w4a16)
            # Memory-bound regime: report bandwidth on the weights (the
            # dominant HBM read). For M small, achieved BW on packed
            # weights is the right metric; for large M, the kernel
            # becomes compute-bound and BW is less meaningful.
            bw = achieved_bandwidth_gbps(packed_bytes, r_w4a16.median_ms)
            marker = " <- WIN" if speedup >= 1.0 else ""
            print(f"    speedup vs fp16: {speedup:.2f}×{marker}    "
                  f"weight BW: {bw:.0f} GB/s")


if __name__ == "__main__":
    main()
