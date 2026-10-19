"""The guarded call path (model-free, a registered test op on CPU): its backward
releases the forward's leaves and outputs while the graph is still referenced, its
gradients equal the plain path's, and a proven op keeps the guarded path until the
step boundary (`settle_guards`)."""

import weakref

import pytest
import torch

from kohakumoss_kernels import registry, status
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OP = "test_guarded_release_square"


class _Square(torch.autograd.Function):
    """x * x with its input saved for the backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return 2 * x * g


@pytest.fixture
def square(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(status, "_CHECKS", {})
    status.clear_failures()
    registry.register(OP, "triton", _Square.apply)
    registry.register(OP, "eager", lambda x: x * x)
    order = registry._STATE.priority["cpu"]
    registry.set_priority("cpu", ("triton", "eager"))
    registry.PROVEN.clear()
    yield
    registry.set_priority("cpu", order)
    registry._STATE.impls.pop(OP, None)
    registry.PROVEN.clear()
    status.clear_failures()


def check_released(op) -> None:
    """After the backward, the guarded node holds none of its forward's leaves or
    outputs although the graph is still referenced; the gradient is the plain one."""
    op.guard = True
    x = torch.randn(64, requires_grad=True)
    out = op(x * 1)
    node = out.grad_fn
    held = [weakref.ref(t) for t in (*node.outs, *node.leaves)]
    root = out.sum()
    root.backward()
    assert torch.allclose(x.grad, 2 * x.detach()), x.grad
    alive = [r for r in held if r() is not None]
    assert not alive, f"{len(alive)} forward tensors outlive the backward"
    assert root.grad_fn is not None


class _Kept(registry._Guarded):
    """The guard keeping its forward state after the backward (the broken variant)."""

    @staticmethod
    def backward(ctx, *grads):
        return (None, None, *registry._backprop(ctx.outs, ctx.leaves, grads))


def test_backward_releases_the_forward_state(square, monkeypatch):
    check_released(registry.resolve(OP, "cpu", torch.float32))
    monkeypatch.setattr(registry, "_Guarded", _Kept)
    op = registry.resolve(OP, "cpu", torch.float32)
    must_fail(lambda: check_released(op), "forward state kept after the backward")


def _path(op) -> str:
    return op._call.__name__


def check_path_fixed_in_step(op) -> None:
    """A good guarded backward leaves the call path guarded until `settle_guards`."""
    op.guard = True
    x = torch.randn(8, requires_grad=True)
    op(x).sum().backward()
    assert _path(op) == "_guarded", "the call path changed inside the step"
    registry.settle_guards()
    assert _path(op) == "_plain"


class _Unguards(registry._Guarded):
    """The guard switching the op to its plain path inside its backward."""

    @staticmethod
    def backward(ctx, *grads):
        got = registry._backprop(ctx.outs, ctx.leaves, grads)
        ctx.op.guard = False
        return (None, None, *got)


def test_call_path_fixed_within_a_step(square, monkeypatch):
    check_path_fixed_in_step(registry.resolve(OP, "cpu", torch.float32))
    monkeypatch.setattr(registry, "_Guarded", _Unguards)
    op = registry.resolve(OP, "cpu", torch.float32)
    must_fail(lambda: check_path_fixed_in_step(op), "unguarded mid-step")
