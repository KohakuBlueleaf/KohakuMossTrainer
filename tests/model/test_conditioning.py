"""Conditioning: fp32 under training autocast, an fp32 sinusoid accurate to fp64, and
16-bit-input / fp32-accumulate / fp32-output linears on a 16-bit trunk."""

import contextlib

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.model.build import build_backbone, resolve_config
from kohakumoss.model.components import wide
from kohakumoss.model.components.embed import Sinusoid
from tests._util.negative import must_fail


def _model(dtype=torch.bfloat16, device="cpu"):
    return build_backbone(resolve_config("tiny"), device=device, dtype=dtype, init=True)


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _check(cond):
    assert cond


@pytest.mark.cpu
def test_conditioning_ignores_autocast():
    model = _model()
    sigma = torch.tensor([0.9, 0.3, 0.05])
    mod, head = model.conditioning(sigma)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mod_ac, head_ac = model.conditioning(sigma)
    assert mod_ac.dtype == torch.float32
    assert torch.equal(mod, mod_ac) and torch.equal(head, head_ac)

    model._fp32_region = contextlib.nullcontext
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mod_raw, _ = model.conditioning(sigma)
    must_fail(lambda: _check(torch.equal(mod, mod_raw)), "autocast lowers modulation")


def _sinusoid_ref(t, freq_dim):
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = t.double()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1)


def _naive(t, freq_dim):
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float32) / half)
    phase = t.float()[:, None] * freq
    return torch.cat((phase.cos(), phase.sin()), -1)


@pytest.mark.cpu
def test_sinusoid_fp32_matches_fp64():
    g = torch.Generator().manual_seed(0)
    grid = torch.arange(1, 1001, dtype=torch.float32)
    t = torch.cat((grid, torch.rand(4000, generator=g) * 1000, torch.tensor([0.0])))
    t = (t / 1000).float() * 1000
    ref = _sinusoid_ref(t, 256)
    out = Sinusoid(256, 1000.0)(t)
    assert out.dtype == torch.float32
    assert (out.double() - ref).abs().max() < 1e-6
    bad = _naive(t, 256)
    must_fail(lambda: _check((bad.double() - ref).abs().max() < 1e-6), "naive fp32")


def _wide_ref(module, x, dtype):
    w = module.weight.to(dtype).double()
    return x.to(dtype).double() @ w.T + module.bias.float().double()


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_conditioning_linears_wide(dtype):
    model = _model(dtype)
    g = torch.Generator().manual_seed(1)
    for module in model.conditioning_linears():
        assert module.weight.dtype == torch.float32 and module.input_dtype == dtype
        x = torch.randn(3, module.in_features, generator=g) * 2
        y = module(x)
        ref = _wide_ref(module, x, dtype)
        assert y.dtype == torch.float32 and _rel(y, ref) < 1e-6
        module.set_input_dtype(None)
        rounded = module(x.to(dtype))
        must_fail(lambda: _check(_rel(rounded, ref) < 1e-6), "16-bit output")
        module.set_input_dtype(dtype)


@pytest.mark.cpu
def test_full_precision_model_keeps_own_dtype():
    model = _model(torch.float32)
    assert all(m.input_dtype is None for m in model.conditioning_linears())


def _grads(fn, x, w, g):
    x = x.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    fn(x, w).backward(g)
    return x.grad, w.grad


def _check_wide_grads(device, dtype):
    gen = torch.Generator().manual_seed(2)
    x = (torch.randn(5, 48, generator=gen) * 3).to(device)
    w = (torch.randn(24, 48, generator=gen) * 0.1).to(device)
    g = torch.randn(5, 24, generator=gen).to(device)
    gx, gw = _grads(lambda a, b: wide.wide_linear(a, b, dtype), x, w, g)
    x16, w16, g16 = (t.to(dtype).double() for t in (x, w, g))
    assert gx.dtype == torch.float32 and gw.dtype == torch.float32
    assert _rel(gx, g16 @ w16) < 1e-6 and _rel(gw, g16.T @ x16) < 1e-6

    def plain(a, b):
        return F.linear(a.to(dtype), b.to(dtype)).float()

    px, _ = _grads(plain, x, w, g)
    must_fail(lambda: _check(_rel(px, g16 @ w16) < 1e-6), "16-bit grads")


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_wide_linear_backward(dtype):
    _check_wide_grads("cpu", dtype)


@pytest.mark.cuda
def test_wide_linear_on_cuda():
    major, _ = torch.cuda.get_device_capability()
    dev = torch.device("cuda")
    assert wide.wide_mm(dev, torch.float16) is wide._mm_native
    expect = wide._mm_native if major >= 8 else wide._mm_rounded
    assert wide.wide_mm(dev, torch.bfloat16) is expect
    for dtype in (torch.float16, torch.bfloat16):
        _check_wide_grads(dev, dtype)


@pytest.mark.xpu
def test_forward_on_xpu():
    model = _model(torch.bfloat16, "xpu")
    mod, head = model.conditioning(torch.tensor([0.5], device="xpu"))
    assert torch.isfinite(mod).all() and torch.isfinite(head).all()
