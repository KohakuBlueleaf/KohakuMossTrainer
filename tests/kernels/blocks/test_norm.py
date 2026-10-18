"""rms_norm / rms_norm_mod backends vs the eager reference (error stats against fp64,
fwd + bwd)."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import norm as eager
from tests._util.negative import must_fail
from tests.kernels.blocks._blockops_harness import check

BACKENDS = ("triton", "tilelang")
CASES = [(3, 200, (17, 1, 40)), (2, 3072, (300, 5)), (1, 128, (33,))]


def _impl(op, backend, device, dtype):
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _inputs(b, d, lens, dtype, device, wdtype=torch.float32):
    n = sum(lens)
    seg = torch.cat(
        [torch.full((k,), i, dtype=torch.int32) for i, k in enumerate(lens)]
    )
    x = (torch.randn(n, d) * 3).to(device, dtype)
    w = (1 + 0.5 * torch.randn(d)).to(device, wdtype)
    mod = per_sample_slots(len(lens), d, device)
    return x, w, mod[:, 1], mod[:, 4], seg.to(device)


def per_sample_slots(b: int, d: int, device, slots: int = 6) -> torch.Tensor:
    """[B, slots, D] fp32 modulation with distinct rows; its `[:, k]` slices are the
    strided per-sample vectors the backbone passes (row stride slots * D)."""
    return (0.3 * torch.randn(b, slots, d)).to(device, torch.float32)


@pytest.mark.gpu
@pytest.mark.budget(30)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", CASES)
def test_rms_norm_mod(backend, dtype, case, device):
    fn = _impl("rms_norm_mod", backend, device, dtype)
    _, d, lens = case
    x, w, shift, scale, seg = _inputs(*case, dtype, device)
    args = (x, w, shift, scale, seg, 1e-6)
    grads = (True, True, True, True, False, False)
    check(fn, eager.rms_norm_mod, args, grads, dtype, f"{backend}-mod")
    args = (x, None, shift, None, seg, 1e-6)
    grads = (True, False, True, False, False, False)
    check(fn, eager.rms_norm_mod, args, grads, dtype, f"{backend}-shift")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm(backend, dtype, device):
    fn = _impl("rms_norm", backend, device, dtype)
    x, w, _, _, _ = _inputs(1, 3072, (257,), dtype, device, wdtype=dtype)
    check(
        fn, eager.rms_norm, (x, w, 1e-6), (True, True, False), dtype, f"{backend}-norm"
    )
    x3 = x.reshape(257, 24, 128)
    w3 = w[:128].contiguous()
    check(
        fn,
        eager.rms_norm,
        (x3, w3, 1e-6),
        (True, True, False),
        dtype,
        f"{backend}-norm3d",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_segment_reduction(backend, device):
    """A backend that sums d_shift over the whole batch instead of per sample fails."""
    fn = _impl("rms_norm_mod", backend, device, torch.float16)

    def broken(x, w, shift, scale, seg, eps):
        whole = shift.mean(0, keepdim=True).expand_as(shift)
        return fn(x, w, whole + (shift - shift.detach()) * 0, scale, seg, eps)

    x, w, shift, scale, seg = _inputs(2, 200, (40, 9), torch.float16, device)
    args = (x, w, shift, scale, seg, 1e-6)
    grads = (True, True, True, True, False, False)
    must_fail(
        lambda: check(broken, eager.rms_norm_mod, args, grads, torch.float16, "broken")
    )
