"""Per-sample modulation in the model's layout: shift / scale / gate arrive as strided
slices `mod[:, k]` of one fp32 [B, 6, D] tensor (`MossBackbone.conditioning`), not as
contiguous [B, D] tensors. Each backend's output and gradients with the strided slices
must equal its own result on contiguous copies of the same values (whose precision
against the eager spec tests/kernels/test_norm.py, test_residual.py and
test_fused_linear.py establish)."""

import pytest
import torch

from kohakumoss_kernels import registry
from tests._util.negative import must_fail
from tests.kernels.test_fused_linear import _force

BACKENDS = ("triton", "tilelang")
LENS = (40, 9, 77)
D = 256
K = 192
TOL = 1e-3


def _impl(op, backend, device, dtype):
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _seg(device) -> torch.Tensor:
    parts = [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(LENS)]
    return torch.cat(parts).to(device)


def _mod(device) -> torch.Tensor:
    return (0.5 * torch.randn(len(LENS), 6, D)).to(device, torch.float32)


def strided(mod: torch.Tensor, k: int) -> torch.Tensor:
    """The model's view: slot k of [B, 6, D], row stride 6 D."""
    return mod[:, k]


def contiguous(mod: torch.Tensor, k: int) -> torch.Tensor:
    """The same values as a contiguous [B, D] copy (the reference layout)."""
    return mod[:, k].contiguous()


def dense_rows(mod: torch.Tensor, k: int) -> torch.Tensor:
    """Slot k's rows read at stride D from its first element (a layout error)."""
    t = mod[:, k]
    return t.as_strided(t.shape, (t.shape[-1], 1))


def _results(run, inputs, view) -> list[torch.Tensor]:
    """Output and every input's gradient of sum(out^2), with `view` slicing `mod`."""
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = run(*leaves, view)
    out.float().square().sum().backward()
    return [out.detach().float()] + [leaf.grad.float() for leaf in leaves]


def assert_layout_free(run, inputs, view=strided) -> None:
    names = ["out"] + [f"grad{i}" for i in range(len(inputs))]
    got = _results(run, inputs, view)
    ref = _results(run, inputs, contiguous)
    for name, a, b in zip(names, got, ref):
        assert torch.isfinite(a).all(), name
        rel = ((a.double() - b.double()).norm() / b.double().norm()).item()
        assert rel < TOL, f"{name}: rel {rel:.3e} between strided and contiguous"


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_mod_strided_modulation(backend, dtype, device):
    fn = _impl("rms_norm_mod", backend, device, dtype)
    seg = _seg(device)

    def run(x, w, mod, view):
        return fn(x, w, view(mod, 0), view(mod, 1), seg, 1e-6)

    x = (3 * torch.randn(sum(LENS), D)).to(device, dtype)
    w = (1 + 0.5 * torch.randn(D)).to(device, torch.float32)
    inputs = (x, w, _mod(device))
    assert_layout_free(run, inputs)
    must_fail(lambda: assert_layout_free(run, inputs, dense_rows), "dense rows")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("r", [1.0, 0.5])
def test_gated_residual_strided_gate(backend, dtype, r, device):
    fn = _impl("gated_residual", backend, device, dtype)
    seg = _seg(device)

    def run(x, o, mod, view):
        return fn(x, o, view(mod, 2), seg, r)

    x = (4 * torch.randn(sum(LENS), D)).to(device, dtype)
    o = torch.randn(sum(LENS), D).to(device, dtype)
    inputs = (x, o, _mod(device))
    assert_layout_free(run, inputs)
    must_fail(lambda: assert_layout_free(run, inputs, dense_rows), "dense rows")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", ("fused", "split"))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_linear_gated_residual_strided_gate(backend, path, dtype, device):
    fn = _impl("linear_gated_residual", backend, device, dtype)
    forced = _force(backend, path)
    seg = _seg(device)

    def run(a, w, x, mod, view):
        return fn(a, w, None, x, view(mod, 5), seg, 1.0)

    try:
        a = torch.randn(sum(LENS), K).to(device, dtype)
        w = (torch.randn(D, K) / K**0.5).to(device, dtype)
        x = (4 * torch.randn(sum(LENS), D)).to(device, dtype)
        inputs = (a, w, x, _mod(device))
        assert_layout_free(run, inputs)
        must_fail(lambda: assert_layout_free(run, inputs, dense_rows), "dense rows")
    finally:
        forced.force_path(None)
