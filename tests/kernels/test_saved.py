"""Block ops save for backward only what their backward uses (frozen weights and
untrained gates keep no activations alive)."""

import pytest
import torch

from kohakumoss_kernels import registry
from tests._util.negative import must_fail

BACKENDS = ("triton", "tilelang")


def _impl(op, backend, device, dtype):
    try:
        fn = registry.resolve(op, device, dtype, backend=backend)
        return getattr(fn, "__wrapped__", fn)
    except registry.NoImplementationError as err:
        pytest.skip(str(err))


def saved_storages(fn, *args) -> set[int]:
    """Storage addresses of every tensor autograd saves while `fn(*args)` runs."""
    seen: set[int] = set()

    def pack(t):
        if t.numel():
            seen.add(t.untyped_storage().data_ptr())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return seen


def _storage(t) -> int:
    return t.untyped_storage().data_ptr()


def _assert_not_saved(t, saved, what):
    assert _storage(t) not in saved, f"{what} is kept for backward"


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_residual_keeps_o_only_for_the_gate_grad(backend, device):
    dtype = torch.float16
    fn = _impl("gated_residual", backend, device, dtype)
    n, d = 300, 256
    x = torch.randn(n, d, device=device).to(dtype).requires_grad_()
    o = torch.randn(n, d, device=device).to(dtype).requires_grad_()
    seg = torch.zeros(n, dtype=torch.int32, device=device)
    gate = torch.randn(1, d, device=device)
    _assert_not_saved(o, saved_storages(fn, x, o, gate, seg, 1.0), "o")
    trained = gate.clone().requires_grad_()
    must_fail(
        lambda: _assert_not_saved(o, saved_storages(fn, x, o, trained, seg, 1.0), "o"),
        "a trained gate needs o",
    )


def _linear_case(device, dtype, m=300, k=256, n=384):
    y = torch.randn(m, k, device=device).to(dtype).requires_grad_()
    w = (torch.randn(n, k, device=device) / k**0.5).to(dtype)
    return y, w


@pytest.mark.gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_fused_linears_keep_no_input_for_frozen_weights(backend, device):
    """With frozen weights the GEMM input is not saved (it only feeds dW)."""
    dtype = torch.float16
    y, w = _linear_case(device, dtype, n=2 * 384)
    situ = _impl("linear_situ", backend, device, dtype)
    _assert_not_saved(y, saved_storages(situ, y, w, None), "situ input")
    a, wd = _linear_case(device, dtype)
    x = torch.randn(300, 384, device=device).to(dtype).requires_grad_()
    seg = torch.zeros(300, dtype=torch.int32, device=device)
    gate = torch.randn(1, 384, device=device)
    res = _impl("linear_gated_residual", backend, device, dtype)
    _assert_not_saved(
        a, saved_storages(res, a, wd, None, x, gate, seg, 1.0), "residual input"
    )
    trained = w.clone().requires_grad_()
    must_fail(
        lambda: _assert_not_saved(y, saved_storages(situ, y, trained, None), "y"),
        "a trained weight needs the input",
    )
