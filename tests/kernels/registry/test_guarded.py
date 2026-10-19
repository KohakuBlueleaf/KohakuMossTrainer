"""`Bound` call paths: bound once per state (recording, guarded, plain) instead of tested
per call; a backward-time fallback recomputes from the version-checked saved inputs."""

import warnings

import pytest
import torch

from kohakumoss_kernels import recording, registry, status
from kohakumoss_kernels.errors import ConfigsFailed
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OP = "test_guarded_square"


class _BadBackward(torch.autograd.Function):
    """x * x forward; a backward that cannot build."""

    @staticmethod
    def forward(ctx, x):
        return x * x

    @staticmethod
    def backward(ctx, g):
        raise ConfigsFailed("bwd_kernel", {"cfg": "no compiler"})


@pytest.fixture
def square(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(status, "_CHECKS", {})
    status.clear_failures()
    registry.register(OP, "triton", _BadBackward.apply)
    registry.register(OP, "eager", lambda x: x * x)
    order = registry._STATE.priority["cpu"]
    registry.set_priority("cpu", ("triton", "eager"))
    yield
    registry.set_priority("cpu", order)
    registry._STATE.impls.pop(OP, None)
    status.clear_failures()


def check_modified_input_refused(op) -> None:
    """A backward fallback after the input was changed in place raises autograd's
    version error instead of recomputing the gradient on the new values."""
    x = torch.full((4,), 3.0, requires_grad=True)
    a = x * 1
    out = op(a)
    a.mul_(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out.sum().backward()
        except RuntimeError as err:
            assert "modified by an inplace operation" in str(err), err
            return
    raise AssertionError(f"recomputed on the modified input: grad {x.grad}")


def test_backward_fallback_reads_checked_inputs(square):
    check_modified_input_refused(registry.resolve(OP, "cpu", torch.float32))


class _Unchecked(torch.autograd.Function):
    """The guard keeping its inputs as plain attributes (no version check)."""

    @staticmethod
    def forward(ctx, op, kwargs, *args):
        leaves = registry._leaves(args)
        with torch.enable_grad():
            out = op.fn(*leaves, **kwargs)
        ctx.op, ctx.kwargs, ctx.args, ctx.leaves, ctx.outs = (
            op,
            kwargs,
            args,
            leaves,
            (out,),
        )
        return out.detach()

    @staticmethod
    def backward(ctx, *grads):
        try:
            got = registry._backprop(ctx.outs, ctx.leaves, grads)
        except registry.CAUGHT as err:
            ctx.op.fall_back(err, ctx.args, ctx.kwargs)
            got = registry._grads(
                ctx.op.fn, ctx.kwargs, registry._leaves(ctx.args), grads
            )
        return (None, None, *got)


def test_negative_unchecked_inputs(square, monkeypatch):
    monkeypatch.setattr(registry, "_Guarded", _Unchecked)
    op = registry.resolve(OP, "cpu", torch.float32)
    must_fail(lambda: check_modified_input_refused(op), "modified input recomputed")


def _path(op) -> str:
    return op._call.__name__


def check_paths(op) -> None:
    """Guarded until a good backward, plain after; recording inside `recording()`."""
    assert _path(op) == "_guarded"
    op.guard = False
    assert _path(op) == "_plain"
    with recording.recording():
        assert _path(op) == "_record"
    assert _path(op) == "_plain"


def test_call_path_rebinds(square):
    check_paths(registry.resolve(OP, "cpu", torch.float32))


def test_negative_recording_not_rebound(square, monkeypatch):
    monkeypatch.setattr(recording, "_CHANGED", [])
    must_fail(lambda: check_paths(registry.resolve(OP, "cpu", torch.float32)), "kept")
