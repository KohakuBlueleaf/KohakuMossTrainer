"""Rotated quantized linear (ConvRot): every registered backend, and every Triton
config (the separate pass and each fused tile), against the eager composition and an
fp64 reference: y = (x H) W^T, dx = (dy W) H, dbias. Forward and input / bias
gradients in fp16 and bf16, under torch.compile(fullgraph) too. No rotation, or the
rotation on the wrong side of the product (the forward output, dy instead of dx), must
fail the check."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from kohakumoss_kernels import registry
from kohakumoss_kernels.errors import BUILD_ERRORS
from kohakumoss_kernels.hadamard import group_hadamard_eager
from kohakumoss_kernels.quant import dequantize
from kohakumoss_kernels.quant.rotated import (
    quant_linear_rotated_eager,
    quant_linear_rotated_reference,
)
from tests._util.error_stats import assert_no_regression
from tests._util.hadamard_ref import rot64
from tests._util.negative import must_fail

SHAPES = (
    (200, 192, 256, 256),
    (513, 384, 1024, 256),
    (1, 96, 512, 256),
    (77, 64, 192, 64),
)


def _quantize(w):
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def _ref64(x, q, scale, bias, group):
    w = q.double() * scale.double()[:, None]
    y = rot64(x, group) @ w.T
    return y if bias is None else y + bias.double()


def _dx64(g, q, scale, group):
    return rot64(g.double() @ (q.double() * scale.double()[:, None]), group)


def _wrong_sides(x, q, scale, bias, group) -> dict:
    """Broken variants: no rotation, and the rotation on the output (N) side of the
    product instead of the input (K) side."""
    w = dequantize(q, scale, torch.float32)
    b = None if bias is None else bias.float()
    plain = F.linear(x.float(), w, b)
    return {
        "no rotation": plain.to(x.dtype),
        "rotation on the output side": group_hadamard_eager(plain, group).to(x.dtype),
    }


def _problem(device, dtype, m, n, k, with_bias):
    w = torch.randn(n, k, device=device) * 0.05
    q, scale = _quantize(w)
    x = torch.randn(m, k, device=device).to(dtype)
    bias = (torch.randn(n, device=device) * 0.1).to(dtype) if with_bias else None
    return x, q, scale, bias


@pytest.mark.cpu
@pytest.mark.parametrize("with_bias", (False, True))
def test_eager_composition_matches_fp64(with_bias):
    x, q, scale, bias = _problem("cpu", torch.float32, 33, 128, 256, with_bias)
    xa = x.clone().requires_grad_(True)
    ba = None if bias is None else bias.clone().requires_grad_(True)
    y = quant_linear_rotated_eager(xa, q, scale, ba, "int8", 64)
    ref = _ref64(x, q, scale, bias, 64)
    torch.testing.assert_close(y.double(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        quant_linear_rotated_reference(x, q, scale, bias, "int8", 64), y
    )
    g = torch.randn_like(y)
    y.backward(g)
    torch.testing.assert_close(
        xa.grad.double(), _dx64(g, q, scale, 64), rtol=1e-5, atol=1e-5
    )
    if with_bias:
        torch.testing.assert_close(
            ba.grad.double(), g.double().sum(0), rtol=1e-5, atol=1e-5
        )
    for what, wrong in _wrong_sides(x, q, scale, bias, 64).items():
        must_fail(
            lambda w=wrong: torch.testing.assert_close(
                w.double(), ref, rtol=1e-3, atol=1e-3
            ),
            what,
        )


def _cases():
    if not torch.cuda.is_available():
        return []
    out = []
    for backend in ("triton", "tilelang"):
        for dtype in (torch.float16, torch.bfloat16):
            why = registry.available(
                "quant_linear_rotated", "cuda", dtype, fmt="int8", group=256
            )
            if why.get(backend) is None:
                out.append((backend, dtype))
    return out


CASES = _cases()


@pytest.mark.cuda
@pytest.mark.parametrize("backend,dtype", CASES)
@pytest.mark.parametrize("m,n,k,group", SHAPES)
@pytest.mark.parametrize("with_bias", (False, True))
def test_forward_backward(backend, dtype, m, n, k, group, with_bias):
    op = registry.resolve(
        "quant_linear_rotated", "cuda", dtype, backend=backend, fmt="int8", group=group
    )
    x, q, scale, bias = _problem("cuda", dtype, m, n, k, with_bias)
    xa = x.clone().requires_grad_(True)
    ba = None if bias is None else bias.clone().requires_grad_(True)
    y = op(xa, q, scale, ba, "int8", group)
    xe = x.clone().requires_grad_(True)
    be = None if bias is None else bias.clone().requires_grad_(True)
    ye = quant_linear_rotated_reference(xe, q, scale, be, "int8", group)
    name = f"{backend}/{dtype}/{m}x{n}x{k}/g{group}"
    assert y.dtype == dtype and y.shape == (m, n)
    ref = _ref64(x, q, scale, bias, group)
    assert_no_regression(y, ye, ref, f"{name} fwd", "quant_rotated")
    g = torch.randn_like(y)
    y.backward(g)
    ye.backward(g)
    assert_no_regression(xa.grad, xe.grad, _dx64(g, q, scale, group), f"{name} dx")
    if with_bias:
        assert_no_regression(ba.grad, be.grad, g.double().sum(0), f"{name} dbias")


def _triton():
    return importlib.import_module("kohakumoss_kernels.backends.triton.quant_rotated")


TRITON = [d for b, d in CASES if b == "triton"]


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", TRITON)
@pytest.mark.parametrize("m,n,k,group", SHAPES)
def test_every_triton_config(dtype, m, n, k, group):
    """Each tuner candidate (separate, every fused tile) on its own; a fused tile the
    card cannot build (shared memory / registers) is skipped, but one must run."""
    mod = _triton()
    x, q, scale, bias = _problem("cuda", dtype, m, n, k, True)
    ye = quant_linear_rotated_reference(x, q, scale, bias, "int8", group)
    g = torch.randn_like(ye)
    dxe = group_hadamard_eager(g @ dequantize(q, scale, dtype), group)
    ref, dref = _ref64(x, q, scale, bias, group), _dx64(g, q, scale, group)
    ran = {"fwd": [], "dgrad": []}
    for cfg in mod.fwd_candidates(k, group):
        try:
            y = mod.fwd_runner(cfg)(x, q, scale, bias, group)
        except BUILD_ERRORS:
            continue
        ran["fwd"].append(cfg)
        assert_no_regression(y, ye, ref, f"{cfg} fwd", "quant_rotated")
    for cfg in mod.dgrad_candidates(k, group):
        try:
            dx = mod.dgrad_runner(cfg)(g, q, scale, group)
        except BUILD_ERRORS:
            continue
        ran["dgrad"].append(cfg)
        assert_no_regression(dx, dxe, dref, f"{cfg} dgrad", "quant_rotated")
    for kind, cfgs in ran.items():
        assert any(c["impl"] == "fused" for c in cfgs), f"no fused {kind} tile ran"
        assert any(c["impl"] == "separate" for c in cfgs), f"separate {kind} failed"


@pytest.mark.cuda
@pytest.mark.parametrize("backend,dtype", CASES)
def test_check_rejects_wrong_sides(backend, dtype):
    m, n, k, group = 64, 256, 512, 256
    x, q, scale, _ = _problem("cuda", dtype, m, n, k, False)
    ye = quant_linear_rotated_reference(x, q, scale, None, "int8", group)
    ref = _ref64(x, q, scale, None, group)
    for what, broken in _wrong_sides(x, q, scale, None, group).items():
        must_fail(lambda b=broken: assert_no_regression(b, ye, ref), what)
    g = torch.randn_like(ye)
    w = dequantize(q, scale, dtype)
    dxe = group_hadamard_eager(g @ w, group)
    early = group_hadamard_eager(g, group) @ w
    must_fail(
        lambda: assert_no_regression(early, dxe, _dx64(g, q, scale, group)),
        "dgrad rotation applied to dy instead of dx",
    )


@pytest.mark.cuda
@pytest.mark.parametrize("backend,dtype", CASES)
def test_fullgraph_compile(backend, dtype):
    group = 256
    op = registry.resolve(
        "quant_linear_rotated", "cuda", dtype, backend=backend, fmt="int8", group=group
    )
    w = torch.randn(128, 512, device="cuda") * 0.05
    q, scale = _quantize(w)
    x = torch.randn(2, 40, 512, device="cuda", dtype=dtype, requires_grad=True)
    fn = torch.compile(
        lambda t: op(t, q, scale, None, "int8", group).float().square().sum(),
        fullgraph=True,
    )
    loss = fn(x)
    loss.backward()
    xe = x.detach().clone().requires_grad_(True)
    ye = quant_linear_rotated_reference(xe, q, scale, None, "int8", group)
    loss_e = ye.float().square().sum()
    loss_e.backward()
    w64 = q.double() * scale.double()[:, None]
    y64 = rot64(x.detach(), group) @ w64.T
    name = f"{backend}/{dtype} compiled"
    assert_no_regression(
        loss, loss_e, y64.square().sum(), f"{name} loss", "quant_rotated"
    )
    dx64 = rot64(2 * y64 @ w64, group)
    assert_no_regression(x.grad, xe.grad, dx64, f"{name} dx", "quant_rotated")
