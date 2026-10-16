"""qk_norm_rope backends vs the eager spec (fp64 bar, fwd + bwd)."""

import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager import rope as eager
from tests._util.negative import must_fail
from tests.kernels._blockops_harness import check

BACKENDS = ("triton", "tilelang")


def _impl(backend, device, dtype):
    try:
        fn = registry.resolve("qk_norm_rope", device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def _inputs(hw, tl_, heads, dh, dtype, device):
    seq = build_seqinfo(torch.tensor(hw), torch.tensor(tl_))
    n = seq.n_joint
    q = (torch.randn(n, heads, dh) * 2).to(device, dtype)
    k = (torch.randn(n, heads, dh) * 2).to(device, dtype)
    qw = (1 + 0.3 * torch.randn(dh)).to(device, torch.float32)
    kw = (1 + 0.3 * torch.randn(dh)).to(device, torch.float32)
    nb = dh // 8
    bands = 1 / (10000.0 ** (torch.arange(nb, dtype=torch.float32) / nb))
    return q, k, qw, kw, seq.pos_joint.to(device), bands.to(device)


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        ([[3, 5], [4, 7]], [7, 300], 3, 128),
        ([[16, 16]], [77], 24, 128),
        ([[2, 3]], [5], 2, 32),
    ],
)
def test_qk_norm_rope(backend, dtype, shape, device):
    fn = _impl(backend, device, dtype)
    q, k, qw, kw, pos, bands = _inputs(*shape, dtype, device)
    args = (q, k, qw, kw, pos, bands, 1e-6)
    check(
        fn,
        eager.qk_norm_rope,
        args,
        (True, True, True, True, False, False, False),
        dtype,
        backend,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_negative_axis_order(backend, device):
    """Swapping the h and w position axes must be caught."""
    fn = _impl(backend, device, torch.float16)
    q, k, qw, kw, pos, bands = _inputs([[3, 5]], [6], 2, 128, torch.float16, device)

    def broken(q, k, qw, kw, pos, bands, eps):
        return fn(q, k, qw, kw, pos[:, [0, 1, 3, 2]].contiguous(), bands, eps)

    args = (q, k, qw, kw, pos, bands, 1e-6)
    grads = (True, True, False, False, False, False, False)
    must_fail(
        lambda: check(broken, eager.qk_norm_rope, args, grads, torch.float16, "broken")
    )
