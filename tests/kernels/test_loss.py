"""token_mse backends vs the eager spec (fp64 bar, fwd + bwd)."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import loss as eager
from tests._util.negative import must_fail
from tests.kernels._blockops_harness import check

BACKENDS = ("triton", "tilelang")


def _impl(backend, device, dtype):
    try:
        fn = registry.resolve("token_mse", device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _inputs(lens, c, dtype, device):
    n = sum(lens)
    seg = torch.cat(
        [torch.full((k,), i, dtype=torch.int32) for i, k in enumerate(lens)]
    )
    pred = torch.randn(n, c).to(device, dtype)
    target = torch.randn(n, c).to(device, dtype)
    weight = torch.rand(len(lens)).to(device, torch.float32)
    mask = (torch.rand(n) > 0.3).float().to(device)
    return pred, target, seg.to(device), weight, mask


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lens", [(17, 1, 40), (4096, 1024)])
def test_token_mse(backend, dtype, lens, device):
    fn = _impl(backend, device, dtype)
    pred, target, seg, weight, mask = _inputs(lens, 128, dtype, device)
    grads = (True, True, False, False, False)
    check(
        fn,
        eager.token_mse,
        (pred, target, seg, weight, mask),
        grads,
        dtype,
        backend,
        factor=3.0,
    )
    check(
        fn,
        eager.token_mse,
        (pred, target, seg, weight, None),
        grads,
        dtype,
        backend,
        factor=3.0,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_weight_ignored(backend, device):
    """Dropping the per-sample weight must be caught."""
    fn = _impl(backend, device, torch.float16)
    pred, target, seg, weight, mask = _inputs((40, 50), 128, torch.float16, device)

    def broken(p, t, s, w, m):
        return fn(p, t, s, torch.ones_like(w), m)

    grads = (True, False, False, False, False)
    must_fail(
        lambda: check(
            broken,
            eager.token_mse,
            (pred, target, seg, weight, mask),
            grads,
            torch.float16,
            "b",
        )
    )
