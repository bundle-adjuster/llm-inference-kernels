"""The LOCKED reference workload — defined once, imported everywhere.

Mirrors the "Reference workload (LOCKED 2026-05-21)" section of
docs/benchmarking-methodology.md. Changing any value here invalidates every
number previously recorded in docs/results/RESULTS.md, so don't.
"""
from __future__ import annotations

# Target model — Llama 3.1 8B Instruct, FP16 base.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# End-to-end serving workload (Phase 0 baselines, Phase 4 integration).
E2E_BATCH = 16
E2E_PROMPT_LEN = 512
E2E_GEN_LEN = 512

# Decode-attention microbenchmark (Phase 1).
ATTN_BATCH = 8
ATTN_KV_LENS = (512, 1024, 2048, 4096, 8192, 16384)

# Quantized-matmul microbenchmark (Phase 3): real Llama linear-layer shapes,
# M sweeping the memory- -> compute-bound boundary.
GEMM_M = (1, 8, 32, 128, 512)

# Llama 3.1 8B head configuration (used by the attention microbenchmark and the
# GQA mapping in the decode kernel).
N_LAYERS = 32
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 128256
