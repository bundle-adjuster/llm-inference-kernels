"""Correctness tests for the fused decode attention kernel.

Activates once the PyTorch extension is built (Phase 1b):
    python setup.py build_ext --inplace
Until then the tests skip cleanly.
"""
import math

import pytest
import torch

from reference.attention_ref import decode_attention

llmik = pytest.importorskip(
    "llmik_cuda",
    reason="build the extension: python setup.py build_ext --inplace")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required")


@requires_cuda
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("seqlen_kv", [128, 2048])
def test_decode_attention_matches_reference(batch, seqlen_kv):
    torch.manual_seed(0)
    n_heads, n_kv_heads, head_dim = 32, 8, 128  # Llama 3 8B head config
    dev = "cuda"
    scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(batch, n_heads, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                    device=dev, dtype=torch.float16)
    v = torch.randn(batch, n_kv_heads, seqlen_kv, head_dim,
                    device=dev, dtype=torch.float16)

    out = llmik.decode_attention(q, k, v, scale)
    ref = decode_attention(q.unsqueeze(2), k, v, scale=scale).squeeze(2)

    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
