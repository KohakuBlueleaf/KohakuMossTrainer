"""Group Hadamard rotation (ConvRot): the matrix against an independent fp64 Kronecker
build, the eager reference against fp64 (fp32 butterflies, 16-bit matmul), its
self-adjoint gradient, and every registered GPU backend (and the compile path) against
eager and fp64 in fp16 / bf16, forward and backward. Broken variants (a Sylvester
Hadamard, an unnormalised one, a wrong group) must fail."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.hadamard import (
    _traced_matrix,
    check_group,
    group_hadamard_eager,
    hadamard_matrix,
    stages,
)
from kohakumoss_kernels.hardware.arch import arch_of
from tests._util.error_stats import assert_no_regression
from tests._util.hadamard_ref import h64, rot64
from tests._util.negative import must_fail

GROUPS = (16, 64, 256)


def close_fp32(got: torch.Tensor, ref: torch.Tensor) -> None:
    """fp32 vector math: within 16 fp32 ulps of the largest reference value."""
    tol = 16 * torch.finfo(torch.float32).eps * ref.abs().max().item()
    err = (got.double() - ref).abs().max().item()
    assert err <= tol, f"max error {err:.3e} > {tol:.3e}"


@pytest.mark.cpu
@pytest.mark.parametrize("group", (4, *GROUPS))
def test_matrix_is_the_normalised_regular_hadamard(group):
    ref = h64(group)
    h = hadamard_matrix(group, torch.float32)
    assert torch.equal(h.double(), ref)
    traced = _traced_matrix(group, stages(group), torch.float32, "cpu")
    assert torch.equal(traced, h), "the in-trace build differs from the cached matrix"
    assert torch.equal(h, h.T)
    eye = torch.eye(group, dtype=torch.float64)
    assert torch.allclose(ref @ ref, eye, atol=1e-12)
    for dtype in (torch.float16, torch.bfloat16):
        assert torch.equal(hadamard_matrix(group, dtype).double(), ref)
    must_fail(lambda: torch.testing.assert_close(h.double(), h64(group, True)))


@pytest.mark.cpu
@pytest.mark.parametrize("group", (0, 1, 2, 8, 12, 32, 128, 255))
def test_non_power_of_4_groups_are_refused(group):
    with pytest.raises(ValueError):
        stages(group)


@pytest.mark.cpu
def test_last_dim_must_divide():
    with pytest.raises(ValueError):
        check_group(torch.zeros(3, 100), 16)
    must_fail(lambda: pytest.raises(ValueError, check_group, torch.zeros(3, 64), 16))


@pytest.mark.cpu
@pytest.mark.parametrize("group", GROUPS)
def test_eager_fp32_butterflies_match_fp64(group):
    x = torch.randn(5, 3, 2 * group)
    ref = rot64(x, group)
    got = group_hadamard_eager(x, group)
    assert got.dtype == torch.float32 and got.shape == x.shape
    close_fp32(got, ref)
    close_fp32(group_hadamard_eager(got, group), x.double())
    must_fail(lambda: close_fp32(got, rot64(x, group, sylvester=True)), "Sylvester")
    must_fail(lambda: close_fp32(got, rot64(x, group, normalise=False)), "unscaled")


def _index_order(x: torch.Tensor, group: int) -> torch.Tensor:
    """The fp32 butterflies with each stage's sum written out in index order."""
    v, rows = x.float(), x.numel() // group
    for a in range(stages(group)):
        v = v.reshape(rows * 4**a, 4, group // 4 ** (a + 1))
        total = ((v[:, 0] + v[:, 1]) + v[:, 2]) + v[:, 3]
        v = torch.stack([total - 2.0 * v[:, 3 - k] for k in range(4)], 1)
    return v.reshape(x.shape) * 2.0 ** -stages(group)


def _pairwise(x: torch.Tensor, group: int) -> torch.Tensor:
    v, rows = x.float(), x.numel() // group
    for a in range(stages(group)):
        v = v.reshape(rows * 4**a, 4, group // 4 ** (a + 1))
        total = (v[:, 0] + v[:, 1]) + (v[:, 2] + v[:, 3])
        v = torch.stack([total - 2.0 * v[:, 3 - k] for k in range(4)], 1)
    return v.reshape(x.shape) * 2.0 ** -stages(group)


def assert_index_order(fn, x: torch.Tensor, group: int) -> None:
    assert torch.equal(fn(x, group), _index_order(x, group))


@pytest.mark.cpu
@pytest.mark.parametrize("group", (4, *GROUPS))
def test_eager_fp32_sums_in_index_order(group):
    """The fp32 butterflies add each 4-point sum in index order, so every device
    rounds them the same (a dequantized weight equals across host and GPU)."""
    x = torch.randn(512, 2 * group) * torch.logspace(-4, 4, 2 * group)
    assert_index_order(group_hadamard_eager, x, group)
    must_fail(lambda: assert_index_order(_pairwise, x, group), "pairwise sums")


@pytest.mark.gpu
@pytest.mark.parametrize("group", (4, *GROUPS))
def test_eager_fp32_equals_across_devices(device, group):
    x = torch.randn(512, 2 * group) * torch.logspace(-4, 4, 2 * group)
    got = group_hadamard_eager(x.to(device), group).cpu()
    assert torch.equal(got, group_hadamard_eager(x, group))


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("group", GROUPS)
def test_eager_half_is_one_rounding_of_fp64(dtype, group):
    """The 16-bit matmul (fp32 accumulation, exact H) rounds the fp64 result once."""
    x = torch.randn(64, group).to(dtype)
    got = group_hadamard_eager(x, group)
    assert got.dtype == dtype
    ref = rot64(x, group)
    ulp = torch.finfo(dtype).eps
    err = ((got.double() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
    assert err <= ulp, f"relative error {err:.3e} > one {dtype} rounding {ulp:.3e}"


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_eager_gradient_is_the_rotation(dtype):
    group = 64
    x = torch.randn(7, 3 * group, dtype=dtype, requires_grad=True)
    g = torch.randn(7, 3 * group, dtype=dtype)
    (dx,) = torch.autograd.grad(group_hadamard_eager(x, group), x, g)
    assert dx.dtype == dtype
    if dtype == torch.float32:
        close_fp32(dx, rot64(g, group))
    else:
        torch.testing.assert_close(dx, group_hadamard_eager(g, group))
    must_fail(lambda: close_fp32(dx, rot64(g, group, sylvester=True)), "Sylvester")


def _gpu_cases():
    dev = None
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        dev = "xpu"
    if dev is None:
        return []
    out = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        if dtype == torch.bfloat16 and not arch_of(dev).has_bf16:
            continue
        avail = registry.available("group_hadamard", dev, dtype)
        if avail.get("triton") is None:
            out.append(("triton", dtype))
        if dtype != torch.float32 and avail.get("compile") is None:
            out.append(("compile", dtype))
    return out


CASES = _gpu_cases()
SHAPES = ((300, 3072), (1, 1024), (1025, 512))


def _impl(backend, device, dtype):
    if backend == "compile":
        return compiled_op("group_hadamard")
    return registry.resolve("group_hadamard", device, dtype, backend=backend)


@pytest.mark.gpu
@pytest.mark.parametrize("backend,dtype", CASES)
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("rows,k", SHAPES)
def test_backend_forward_backward(device, backend, dtype, group, rows, k):
    op = _impl(backend, device, dtype)
    x = torch.randn(rows, k, device=device).to(dtype)
    xa = x.clone().requires_grad_(True)
    xe = x.clone().requires_grad_(True)
    y = op(xa, group)
    ye = group_hadamard_eager(xe, group)
    name = f"{backend}/{dtype}/g{group}/{rows}x{k}"
    assert y.dtype == dtype and y.shape == x.shape
    assert_no_regression(y, ye, rot64(x, group), f"{name} fwd", "hadamard")
    g = torch.randn_like(y)
    y.backward(g)
    ye.backward(g)
    assert_no_regression(xa.grad, xe.grad, rot64(g, group), f"{name} bwd", "hadamard")


@pytest.mark.gpu
@pytest.mark.parametrize("backend,dtype", CASES)
def test_backend_check_rejects_sylvester(device, backend, dtype):
    group, x = 256, torch.randn(64, 512, device=device).to(dtype)
    ye = group_hadamard_eager(x, group)
    broken = rot64(x, group, sylvester=True).to(dtype)
    must_fail(lambda: assert_no_regression(broken, ye, rot64(x, group)), "Sylvester")
    unscaled = rot64(x, group, normalise=False).to(dtype)
    must_fail(lambda: assert_no_regression(unscaled, ye, rot64(x, group)), "unscaled")


@pytest.mark.gpu
@pytest.mark.parametrize("backend,dtype", CASES)
def test_fullgraph_compile(device, backend, dtype):
    op = _impl(backend, device, dtype)
    x = torch.randn(2, 40, 512, device=device).to(dtype).requires_grad_(True)
    fn = torch.compile(lambda t: op(t, 256).float().square().sum(), fullgraph=True)
    loss = fn(x)
    loss.backward()
    xe = x.detach().clone().requires_grad_(True)
    loss_e = group_hadamard_eager(xe, 256).float().square().sum()
    loss_e.backward()
    y64 = rot64(x.detach(), 256)
    name = f"{backend}/{dtype} compiled"
    assert_no_regression(loss, loss_e, y64.square().sum(), f"{name} loss", "hadamard")
    assert_no_regression(x.grad, xe.grad, rot64(2 * y64, 256), f"{name} dx", "hadamard")
