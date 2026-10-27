"""int8 compute linear: the eager reference against an fp64 model of the same scheme
(rotate, per-row int8 activations, exact integer products, both scales), every
registered backend and every Triton GEMM config against the eager reference and fp64,
forward and both input-gradient modes, in fp16 and bf16, under torch.compile too. A
missing rotation, a missing activation scale, the weight scales on the wrong axis of
the input gradient and an unquantized activation must fail the checks."""

import importlib

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.errors import BUILD_ERRORS
from kohakumoss_kernels.int8 import (
    dgrad_reference,
    fwd_reference,
    int8_linear_eager,
    quantize_rows,
)
from kohakumoss_kernels.quant import dequantize
from tests._util.error_stats import assert_no_regression
from tests._util.hadamard_ref import rot64
from tests._util.negative import must_fail

SHAPES = (
    (200, 256, 256, 256),
    (513, 384, 1024, 256),
    (5, 512, 768, 256),
    (77, 192, 256, 0),
    (61, 8192, 4096, 256),
)


def _quantize_weight(w):
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.float()


def _rows64(x):
    """Per-row int8 codes and scales in fp64 (the scheme, exact)."""
    s = x.abs().amax(1).clamp_min(1e-12) / 127.0
    return torch.round(x / s[:, None]).clamp(-127, 127), s


def _rot(x, group):
    return rot64(x, group) if group else x.double()


def _ref64(x, q, scale, bias, group):
    xq, xs = _rows64(_rot(x, group))
    y = (xq @ q.double().T) * xs[:, None] * scale.double()[None, :]
    return y if bias is None else y + bias.double()


def _dx64(g, q, scale, group, mode):
    if mode == "16bit":
        return _rot(g.double() @ (q.double() * scale.double()[:, None]), group)
    gq, gs = _rows64(g.double() * scale.double()[None, :])
    return _rot((gq @ q.double()) * gs[:, None], group)


def _problem(device, dtype, m, n, k, with_bias):
    w = torch.randn(n, k, device=device) * 0.05
    q, scale = _quantize_weight(w)
    x = torch.randn(m, k, device=device).to(dtype)
    bias = (torch.randn(n, device=device) * 0.1).to(dtype) if with_bias else None
    return x, q, scale, bias


def _broken_fwd(x, q, scale, bias, group) -> dict:
    """Forwards of broken designs: no rotation, the activation scale dropped, the
    activation left in 16 bits."""
    xq0, xs0 = quantize_rows(x)
    plain = (xq0.double() @ q.double().T) * xs0.double()[:, None] * scale.double()
    rq, rs = quantize_rows(_rot(x, group).to(x.dtype))
    unscaled = (rq.double() @ q.double().T) * scale.double()
    dense = _rot(x, group) @ (q.double() * scale.double()[:, None]).T
    out = {"activation scale dropped": unscaled, "activation not quantized": dense}
    if group:
        out["no rotation"] = plain
    b = 0 if bias is None else bias.double()
    return {what: (y + b).to(x.dtype) for what, y in out.items()}


def _on_grid(m, k, group):
    """x [m, k] fp32 whose rotation x R is exactly codes / 64 with a +-127 code per row:
    every quantization is exact, so the eager result must equal the fp64 model."""
    codes = torch.randint(-126, 127, (m, k)).double()
    codes[torch.arange(m), torch.randint(0, k, (m,))] = 127.0
    return _rot(codes / 64.0, group).float()


@pytest.mark.cpu
@pytest.mark.parametrize("group", (0, 64))
@pytest.mark.parametrize("mode", ("int8", "16bit"))
def test_eager_matches_fp64(group, mode):
    _, q, scale, bias = _problem("cpu", torch.float32, 33, 128, 256, True)
    x = _on_grid(33, 256, group)
    xa = x.clone().requires_grad_(True)
    ba = bias.clone().requires_grad_(True)
    y = int8_linear_eager(xa, q, scale, ba, group, mode)
    ref = _ref64(x, q, scale, bias, group)
    torch.testing.assert_close(y.double(), ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fwd_reference(x, q, scale, bias, group), y)
    g = torch.randn_like(y)
    y.backward(g)
    dref = _dx64(g, q, scale, group, mode)
    torch.testing.assert_close(xa.grad.double(), dref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        ba.grad.double(), g.double().sum(0), rtol=1e-5, atol=1e-5
    )
    if mode == "int8":
        wrong = (
            dgrad_reference(g, q, torch.ones_like(scale), group, mode) * scale.mean()
        )
        must_fail(
            lambda: torch.testing.assert_close(
                wrong.double(), dref, rtol=1e-3, atol=1e-3
            ),
            "weight scales not moved onto dy's columns",
        )


@pytest.mark.cpu
@pytest.mark.parametrize("group", (0, 64))
def test_eager_check_rejects_broken_designs(group):
    """Off the grid (random x) the eager result is within one code flip of the fp64
    model; each broken design is far outside that."""
    x, q, scale, bias = _problem("cpu", torch.float32, 33, 128, 256, True)
    ref = _ref64(x, q, scale, bias, group)

    def close(y):
        torch.testing.assert_close(y.double(), ref, rtol=1e-3, atol=1e-3)

    close(fwd_reference(x, q, scale, bias, group))
    for what, wrong in _broken_fwd(x, q, scale, bias, group).items():
        must_fail(lambda w=wrong: close(w), what)


@pytest.mark.cpu
def test_quantize_rows_rounds_half_even_and_clamps():
    x = torch.tensor([[127.0, 0.5, 1.5, -2.5, -127.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    q, s = quantize_rows(x)
    assert s.tolist() == [1.0, pytest.approx(1e-12 / 127.0)]
    assert q[0].tolist() == [127, 0, 2, -2, -127]
    assert q[1].tolist() == [0] * 5


def _cases():
    if not torch.cuda.is_available():
        return []
    out = []
    for dtype in (torch.float16, torch.bfloat16):
        why = registry.available("int8_linear", "cuda", dtype, group=256, dgrad="int8")
        if why.get("triton") is None:
            out.append(dtype)
    return out


CASES = _cases()


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", CASES)
@pytest.mark.parametrize("m,n,k,group", SHAPES)
@pytest.mark.parametrize("mode", ("int8", "16bit"))
def test_triton_forward_backward(dtype, m, n, k, group, mode):
    op = registry.resolve(
        "int8_linear", "cuda", dtype, backend="triton", group=group, dgrad=mode
    )
    x, q, scale, bias = _problem("cuda", dtype, m, n, k, True)
    xa = x.clone().requires_grad_(True)
    y = op(xa, q, scale, bias, group, mode)
    ye = fwd_reference(x, q, scale, bias, group)
    name = f"{dtype}/{m}x{n}x{k}/g{group}/{mode}"
    assert y.dtype == dtype and y.shape == (m, n)
    ref = _ref64(x, q, scale, bias, group)
    assert_no_regression(y, ye, ref, f"{name} fwd", "int8_linear")
    g = torch.randn_like(y)
    y.backward(g)
    dxe = dgrad_reference(g, q, scale, group, mode)
    dref = _dx64(g, q, scale, group, mode)
    assert_no_regression(xa.grad, dxe, dref, f"{name} dx", "int8_linear")


def _triton():
    return importlib.import_module("kohakumoss_kernels.backends.triton.int8")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", CASES)
@pytest.mark.parametrize("m,n,k,group", SHAPES)
def test_every_gemm_config(dtype, m, n, k, group):
    """Each forward GEMM candidate (cuBLAS, every fused tile the card has) on its own;
    a tile the card cannot build is skipped, but cuBLAS must run."""
    mod = _triton()
    x, q, scale, bias = _problem("cuda", dtype, m, n, k, True)
    xq, xs = mod.quant_rows(x, None, group)
    q64, s64 = _rows64(_rot(x, group))
    assert (xq.double() - q64).abs().max() <= 1
    torch.testing.assert_close(xs.double(), s64, rtol=1e-5, atol=0)
    ye = fwd_reference(x, q, scale, bias, group)
    ref = _ref64(x, q, scale, bias, group)
    ran = []
    for cfg in mod.fused_candidates(x.device, k):
        try:
            y = mod.gemm_runner(cfg)(xq, xs, q, scale, bias, dtype)
        except BUILD_ERRORS:
            continue
        ran.append(cfg["impl"])
        assert_no_regression(y, ye, ref, f"{cfg} fwd", "int8_linear")
    assert "cublas" in ran


def rot_dtype(x, group):
    return rot64(x, group).to(x.dtype)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", CASES)
def test_check_rejects_broken_designs(dtype):
    m, n, k, group = 64, 256, 512, 256
    x, q, scale, bias = _problem("cuda", dtype, m, n, k, True)
    ye = fwd_reference(x, q, scale, bias, group)
    ref = _ref64(x, q, scale, bias, group)
    for what, broken in _broken_fwd(x, q, scale, bias, group).items():
        if what == "activation not quantized":
            continue
        must_fail(lambda b=broken: assert_no_regression(b, ye, ref), what)
    g = torch.randn_like(ye)
    dxe = dgrad_reference(g, q, scale, group, "int8")
    dref = _dx64(g, q, scale, group, "int8")
    unscaled = dgrad_reference(g, q, torch.ones_like(scale), group, "int8")
    must_fail(
        lambda: assert_no_regression(unscaled, dxe, dref),
        "weight scales not moved onto dy's columns",
    )
    early = dgrad_reference(g, q, scale, 0, "int8")
    must_fail(
        lambda: assert_no_regression(early, dxe, dref), "input gradient not rotated"
    )


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", CASES)
def test_fullgraph_compile(dtype):
    group = 256
    op = registry.resolve(
        "int8_linear", "cuda", dtype, backend="triton", group=group, dgrad="int8"
    )
    w = torch.randn(256, 512, device="cuda") * 0.05
    q, scale = _quantize_weight(w)
    x = torch.randn(2, 40, 512, device="cuda", dtype=dtype, requires_grad=True)
    fn = torch.compile(
        lambda t: op(t, q, scale, None, group, "int8").float().square().sum(),
        fullgraph=True,
    )
    loss = fn(x)
    loss.backward()
    xe = x.detach().clone().requires_grad_(True)
    loss_e = int8_linear_eager(xe, q, scale, None, group, "int8").float().square().sum()
    loss_e.backward()
    x2 = x.detach().reshape(-1, 512)
    y64 = _ref64(x2, q, scale, None, group)
    assert_no_regression(
        loss, loss_e, y64.square().sum(), "compiled loss", "int8_linear"
    )
    dx64 = _dx64((2 * y64).to(dtype), q, scale, group, "int8").reshape(x.shape)
    assert_no_regression(x.grad, xe.grad, dx64, "compiled dx", "int8_linear")


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", CASES)
def test_weightless_base_runs(dtype):
    """A stride-0 (weightless) base, as the shape twin binds it, runs every path."""
    op = registry.resolve(
        "int8_linear", "cuda", dtype, backend="triton", group=256, dgrad="int8"
    )
    q = torch.zeros((), dtype=torch.int8, device="cuda").expand(256, 512)
    scale = torch.zeros((), device="cuda").expand(256)
    x = torch.randn(40, 512, device="cuda", dtype=dtype, requires_grad=True)
    y = op(x, q, scale, None, 256, "int8")
    y.float().sum().backward()
    assert y.abs().max() == 0 and x.grad.abs().max() == 0
    ye = fwd_reference(x.detach(), q, scale, None, 256)
    assert ye.abs().max() == 0
    assert dgrad_reference(y.detach(), q, scale, 256, "int8").abs().max() == 0
    assert dequantize(q, scale, dtype).shape == (256, 512)
