"""gated_residual backends vs the eager reference (error stats against fp64, fwd +
bwd)."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import norm as eager
from tests._util.negative import must_fail
from tests.kernels.blocks._blockops_harness import check

BACKENDS = ("triton", "tilelang")


def _impl(backend, device, dtype):
    try:
        fn = registry.resolve("gated_residual", device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _inputs(lens, d, dtype, device):
    n = sum(lens)
    seg = torch.cat(
        [torch.full((k,), i, dtype=torch.int32) for i, k in enumerate(lens)]
    )
    x = (torch.randn(n, d) * 4).to(device, dtype)
    o = torch.randn(n, d).to(device, dtype)
    gate = (0.5 * torch.randn(len(lens), 6, d)).to(device, torch.float32)[:, 2]
    return x, o, gate, seg.to(device)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("r", [1.0, 0.5])
@pytest.mark.parametrize("case", [((17, 1, 40), 200), ((300, 5), 3072)])
def test_gated_residual(backend, dtype, r, case, device):
    fn = _impl(backend, device, dtype)
    x, o, gate, seg = _inputs(*case, dtype, device)
    check(
        fn,
        eager.gated_residual,
        (x, o, gate, seg, r),
        (True, True, True, False, False),
        dtype,
        backend,
    )
    check(
        fn,
        eager.gated_residual,
        (x, o, gate, seg, r),
        (True, True, False, False, False),
        dtype,
        f"{backend}-frozen-gate",
    )
    check(
        fn,
        eager.gated_residual,
        (x, o, None, None, r),
        (True, True, False, False, False),
        dtype,
        backend,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_gate_read_as_contiguous(backend, device):
    """A kernel that indexes a strided gate as row-major [B, D] must be caught."""
    fn = _impl(backend, device, torch.float16)
    x, o, gate, seg = _inputs((20, 30, 7), 256, torch.float16, device)

    def broken(x, o, gate, seg, r):
        return fn(x, o, gate.as_strided(gate.shape, (gate.shape[1], 1)), seg, r)

    grads = (True, True, False, False, False)
    args = (x, o, gate, seg, 1.0)
    must_fail(
        lambda: check(broken, eager.gated_residual, args, grads, torch.float16, "b")
    )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_gate_sample(backend, device):
    """Using sample 0's gate for every row must be caught."""
    fn = _impl(backend, device, torch.float16)
    x, o, gate, seg = _inputs((20, 30), 256, torch.float16, device)

    def broken(x, o, gate, seg, r):
        return fn(x, o, gate, torch.zeros_like(seg), r)

    grads = (True, True, True, False, False)
    must_fail(
        lambda: check(
            broken,
            eager.gated_residual,
            (x, o, gate, seg, 1.0),
            grads,
            torch.float16,
            "b",
        )
    )
