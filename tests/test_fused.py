"""Correctness tests for the fused decode-path elementwise kernels (Phase 10):
fused RMSNorm, SwiGLU (silu*mul), and RoPE, each vs a PyTorch reference.
"""
import pytest
import torch
import torch.nn.functional as F

llmik_cuda = pytest.importorskip("llmik_cuda")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")


def _rms_ref(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (w * (xf * torch.rsqrt(var + eps)).to(x.dtype))


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


@pytest.mark.parametrize("shape", [(16, 4096), (16, 1024), (1, 4096), (512, 4096)])
def test_rmsnorm(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
    out = llmik_cuda.rmsnorm(x, w, 1e-5)
    ref = _rms_ref(x, w, 1e-5)
    rel = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).mean()
    assert rel < 2e-2, rel


def test_rmsnorm_3d():
    torch.manual_seed(0)
    x = torch.randn(4, 128, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    out = llmik_cuda.rmsnorm(x, w, 1e-5)
    assert out.shape == x.shape
    assert torch.allclose(out.float(), _rms_ref(x, w, 1e-5).float(), atol=3e-2)


@pytest.mark.parametrize("shape", [(16, 14336), (1, 14336), (512, 4096)])
def test_silu_mul(shape):
    torch.manual_seed(0)
    g = torch.randn(*shape, device="cuda", dtype=torch.float16)
    u = torch.randn(*shape, device="cuda", dtype=torch.float16)
    out = llmik_cuda.silu_mul(g, u)
    ref = F.silu(g.float()) * u.float()
    assert torch.allclose(out.float(), ref, atol=2e-2), (out.float() - ref).abs().max()


@pytest.mark.parametrize("B,H,S,D", [(16, 32, 1, 128), (16, 8, 1, 128), (16, 32, 512, 128)])
def test_rope(B, H, S, D):
    torch.manual_seed(0)
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    cos = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
    sin = torch.randn(B, S, D, device="cuda", dtype=torch.float16)
    out = llmik_cuda.rope(x, cos, sin)
    ref = (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))
    assert torch.allclose(out.float(), ref.float(), atol=2e-2), (out.float() - ref.float()).abs().max()


def test_rope_broadcast_cos():
    # cos/sin provided as [S, D] (no batch) must broadcast over batch.
    torch.manual_seed(0)
    B, H, S, D = 8, 32, 1, 128
    x = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    cos = torch.randn(S, D, device="cuda", dtype=torch.float16)
    sin = torch.randn(S, D, device="cuda", dtype=torch.float16)
    out = llmik_cuda.rope(x, cos, sin)
    ref = (x * cos[None, None]) + (_rotate_half(x) * sin[None, None])
    assert torch.allclose(out.float(), ref.float(), atol=2e-2)
