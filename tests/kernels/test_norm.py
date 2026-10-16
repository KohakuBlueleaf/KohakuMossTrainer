"""rms_norm / rms_norm_mod backends vs the eager spec (fp64 bar, fwd + bwd)."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import norm as eager
from tests._util.negative import must_fail
from tests.kernels._blockops_harness import check

BACKENDS = ("triton", "tilelang")
FP32_FACTOR = 3.0
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
    shift = (0.3 * torch.randn(len(lens), d)).to(device, torch.float32)
    scale = (0.3 * torch.randn(len(lens), d)).to(device, torch.float32)
    return x, w, shift, scale, seg.to(device)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("case", CASES)
def test_rms_norm_mod(backend, dtype, case, device):
    fn = _impl("rms_norm_mod", backend, device, dtype)
    _, d, lens = case
    x, w, shift, scale, seg = _inputs(*case, dtype, device)
    factor = FP32_FACTOR if dtype == torch.float32 else 1.1
    args = (x, w, shift, scale, seg, 1e-6)
    grads = (True, True, True, True, False, False)
    check(fn, eager.rms_norm_mod, args, grads, dtype, f"{backend}-mod", factor=factor)
    args = (x, None, shift, None, seg, 1e-6)
    grads = (True, False, True, False, False, False)
    check(fn, eager.rms_norm_mod, args, grads, dtype, f"{backend}-shift", factor=factor)


@pytest.mark.cuda
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


@pytest.mark.cuda
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
