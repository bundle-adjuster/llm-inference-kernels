"""Benchmarking + correctness harness shared by all benchmarks and tests.

Implements the protocol in docs/benchmarking-methodology.md: CUDA-event timing,
warmup, median + p10/p90, correctness gating, memory and bandwidth accounting.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class TimingResult:
    """One kernel's timing summary across `iters` measured runs.

    Times are in milliseconds; `p10`/`p90` are the 10th/90th percentile
    samples (a cheap stability indicator alongside the median).
    """

    name: str
    median_ms: float
    p10_ms: float
    p90_ms: float
    iters: int

    def __str__(self) -> str:
        return (f"{self.name:<34} {self.median_ms:9.4f} ms  "
                f"(p10 {self.p10_ms:.4f} / p90 {self.p90_ms:.4f}, n={self.iters})")


def benchmark(
    fn: Callable[[], object],
    *,
    name: str = "kernel",
    warmup: int = 25,
    iters: int = 100,
) -> TimingResult:
    """Time a no-arg callable with CUDA events and return a `TimingResult`.

    Args:
        fn: a no-argument callable. Pass closures (`lambda: kernel(...)`) for
            anything that needs arguments. Each call must launch CUDA work
            on the current stream — the timer measures GPU time between
            `start.record()` and `end.record()`.
        name: label printed in the result.
        warmup: untimed calls to amortise JIT compile / first-touch costs.
        iters: timed calls used to compute median / p10 / p90.

    Returns:
        TimingResult with median / p10 / p90 in milliseconds.

    Raises:
        RuntimeError: if CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for benchmarking")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))  # milliseconds

    samples.sort()
    return TimingResult(
        name=name,
        median_ms=statistics.median(samples),
        p10_ms=samples[max(0, int(0.10 * len(samples)) - 1)],
        p90_ms=samples[min(len(samples) - 1, int(0.90 * len(samples)))],
        iters=iters,
    )


def check_close(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str = "kernel",
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> float:
    """Gate performance on correctness. Raises if the kernel disagrees.

    Both tensors are upcast to fp32 before comparison so fp16 round-off
    doesn't trigger spurious failures. The max absolute diff is printed
    alongside a PASS/FAIL marker (useful for spotting changes that are
    "still within tolerance, but barely").

    Args:
        actual: kernel output, any dtype broadcastable to fp32.
        reference: oracle output, any dtype broadcastable to fp32.
        name: label printed alongside the diff.
        rtol, atol: relative / absolute tolerance, forwarded to
            `torch.allclose`.

    Returns:
        the max absolute diff observed (useful for logging).

    Raises:
        AssertionError: if `torch.allclose` is False.
    """
    torch.cuda.synchronize()
    max_abs = (actual.float() - reference.float()).abs().max().item()
    ok = torch.allclose(actual.float(), reference.float(), rtol=rtol, atol=atol)
    print(f"{name}: max|abs diff| = {max_abs:.3e}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(
            f"{name} mismatch vs reference: max abs diff {max_abs:.3e} "
            f"(rtol={rtol}, atol={atol})")
    return max_abs


def peak_memory_mb(fn: Callable[[], object]) -> float:
    """Peak CUDA memory (MiB) allocated while running `fn`.

    Resets `torch.cuda.max_memory_allocated()` before the call and
    `cudaSynchronize`s after, so the number reflects this call only.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def achieved_bandwidth_gbps(bytes_moved: int, median_ms: float) -> float:
    """Effective HBM bandwidth in GB/s. Compare to peak in `env-report.md`.

    `bytes_moved` should reflect *logical* traffic — one read of K + one
    read of V for a decode-attention call, for example — not raw HBM bytes
    (which depend on L1/L2 behaviour and are best read off `ncu`).
    """
    return bytes_moved / (median_ms * 1e-3) / 1e9
