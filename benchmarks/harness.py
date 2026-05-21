"""Benchmarking + correctness harness shared by all benchmarks and tests.

Implements the protocol in docs/benchmarking-methodology.md: CUDA-event timing,
warmup, median + p10/p90, correctness gating, memory and bandwidth accounting.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch


@dataclass
class TimingResult:
    name: str
    median_ms: float
    p10_ms: float
    p90_ms: float
    iters: int

    def __str__(self) -> str:
        return (f"{self.name:<34} {self.median_ms:9.4f} ms  "
                f"(p10 {self.p10_ms:.4f} / p90 {self.p90_ms:.4f}, n={self.iters})")


def benchmark(fn, *, name="kernel", warmup=25, iters=100) -> TimingResult:
    """Time a no-arg callable with CUDA events. Pass closures for args."""
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


def check_close(actual, reference, *, name="kernel", rtol=2e-2, atol=2e-2):
    """Gate performance on correctness. Raises if the kernel disagrees."""
    torch.cuda.synchronize()
    max_abs = (actual.float() - reference.float()).abs().max().item()
    ok = torch.allclose(actual.float(), reference.float(), rtol=rtol, atol=atol)
    print(f"{name}: max|abs diff| = {max_abs:.3e}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(
            f"{name} mismatch vs reference: max abs diff {max_abs:.3e} "
            f"(rtol={rtol}, atol={atol})")
    return max_abs


def peak_memory_mb(fn) -> float:
    """Peak CUDA memory (MB) allocated while running fn."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def achieved_bandwidth_gbps(bytes_moved: int, median_ms: float) -> float:
    """Effective HBM bandwidth (GB/s). Compare to peak from env-report.md."""
    return bytes_moved / (median_ms * 1e-3) / 1e9
