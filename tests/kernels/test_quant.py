"""Quantized frozen-base linear: every registered backend against an fp64 reference.

Forward and input/bias gradients in fp16 (and bf16 where the arch has it), int8
everywhere and fp8 from sm89, off-tile shapes, under torch.compile(fullgraph).
"""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.quant import quant_linear_reference
from tests._util.negative import must_fail
from tests._util.precision import assert_bar

pytestmark = pytest.mark.cuda

SHAPES = ((200, 150, 136), (1, 96, 64), (513, 384, 1000))


def _quantize(w, fmt):
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    if fmt == "int8":
        scale = amax / 127.0
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    else:
        scale = amax / 448.0
        q = (w / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale.float()


def _ref64(x, q, scale, bias):
    w = q.double() * scale.double()[:, None]
    y = x.double() @ w.T
    return y if bias is None else y + bias.double()


def _cases():
    if not torch.cuda.is_available():
        return []
    arch = arch_of("cuda")
    out = []
    for backend in ("triton", "tilelang"):
        for fmt in ("int8", "fp8"):
            for dtype in (torch.float16, torch.bfloat16):
                reason = registry.available("quant_linear", "cuda", dtype, fmt=fmt)
                if reason.get(backend) is None and (
                    dtype != torch.bfloat16 or arch.has_bf16
                ):
                    out.append((backend, fmt, dtype))
    return out


CASES = _cases()


@pytest.mark.parametrize("backend,fmt,dtype", CASES)
@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("with_bias", (False, True))
def test_forward_backward(backend, fmt, dtype, m, n, k, with_bias):
    torch.manual_seed(0)
    op = registry.resolve("quant_linear", "cuda", dtype, backend=backend, fmt=fmt)
    w = torch.randn(n, k, device="cuda") * 0.05
    q, scale = _quantize(w, fmt)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    bias = (torch.randn(n, device="cuda") * 0.1).to(dtype) if with_bias else None

    xa = x.clone().requires_grad_(True)
    ba = None if bias is None else bias.clone().requires_grad_(True)
    y = op(xa, q, scale, ba, fmt)
    xe = x.clone().requires_grad_(True)
    be = None if bias is None else bias.clone().requires_grad_(True)
    ye = quant_linear_reference(xe, q, scale, be, fmt)
    ref = _ref64(x, q, scale, bias)
    assert y.dtype == dtype and y.shape == (m, n)
    assert_bar(y, ye, ref, dtype, name=f"{backend}/{fmt} fwd")

    g = torch.randn_like(y)
    y.backward(g)
    ye.backward(g)
    w64 = q.double() * scale.double()[:, None]
    assert_bar(xa.grad, xe.grad, g.double() @ w64, dtype, name=f"{backend}/{fmt} dx")
    if with_bias:
        assert_bar(ba.grad, be.grad, g.double().sum(0), dtype, name="dbias")


@pytest.mark.parametrize("backend,fmt,dtype", CASES)
def test_bar_rejects_wrong_scale_axis(backend, fmt, dtype):
    torch.manual_seed(0)
    m, n, k = 64, 96, 96
    w = torch.randn(n, k, device="cuda") * 0.05
    q, scale = _quantize(w, fmt)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    ye = quant_linear_reference(x, q, scale, None, fmt)
    scale_in = scale[torch.arange(k, device="cuda") % n]
    broken = (x.float() @ (q.float() * scale_in[None, :]).T).to(dtype)

    def check():
        assert_bar(broken, ye, _ref64(x, q, scale, None), dtype)

    must_fail(check, "scale applied along the input axis")


@pytest.mark.parametrize("backend,fmt,dtype", CASES)
def test_fullgraph_compile(backend, fmt, dtype):
    torch.manual_seed(0)
    op = registry.resolve("quant_linear", "cuda", dtype, backend=backend, fmt=fmt)
    w = torch.randn(128, 192, device="cuda") * 0.05
    q, scale = _quantize(w, fmt)
    x = torch.randn(2, 40, 192, device="cuda", dtype=dtype, requires_grad=True)
    fn = torch.compile(
        lambda t: op(t, q, scale, None, fmt).float().square().sum(), fullgraph=True
    )
    loss = fn(x)
    loss.backward()
    xe = x.detach().clone().requires_grad_(True)
    loss_e = op(xe, q, scale, None, fmt).float().square().sum()
    loss_e.backward()
    torch.testing.assert_close(loss, loss_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x.grad, xe.grad, rtol=1e-2, atol=1e-3)
