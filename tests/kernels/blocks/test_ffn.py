"""situ_glu backends vs the eager reference (error stats against fp64, fwd + bwd)."""

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import ffn as eager
from tests._util.negative import must_fail
from tests.kernels.blocks._blockops_harness import check

BACKENDS = ("triton", "tilelang")


def _impl(backend, device, dtype):
    try:
        fn = registry.resolve("situ_glu", device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(37, 2 * 200), (513, 2 * 8192), (3, 4, 2 * 64)])
def test_situ_glu(backend, dtype, shape, device):
    fn = _impl(backend, device, dtype)
    x = (torch.randn(*shape) * 8).to(device, dtype)
    check(fn, eager.situ_glu, (x,), (True,), dtype, backend)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_halves_swapped(backend, device):
    """Reading [gate | up] instead of [up | gate] must be caught."""
    fn = _impl(backend, device, torch.float16)

    def broken(x):
        u, g = x.chunk(2, -1)
        return fn(torch.cat((g, u), -1))

    x = (torch.randn(64, 2 * 256) * 8).to(device, torch.float16)
    must_fail(
        lambda: check(broken, eager.situ_glu, (x,), (True,), torch.float16, "broken")
    )
