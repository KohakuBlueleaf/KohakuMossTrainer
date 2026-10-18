"""Prewarm replay of recorded `Bound` calls: each replays under the autocast state it
was recorded under, and a backend failing in the replay rebinds the op (and the model's
op of the same query) before step 1."""

import logging

import pytest
import torch

from kohakumoss_kernels import recording, registry, status
from kohakumoss_kernels.errors import BackendUnavailable
from kohakumoss_kernels.recording import Call
from kohakumoss_kernels.tuning import replay, tuner
from tests._util.logs import collecting
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

OP = "test_replay_probe"


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """`OP` on cpu, priority triton -> eager; the triton impl notes the autocast state
    it runs under and raises `fail[0]` while `fail` is set."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(status, "_CHECKS", {})
    status.clear_failures()
    tuner._TABLES.clear()
    seen, fail = [], []

    def kernel(x):
        if fail:
            raise fail[0]
        seen.append((torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu")))
        return x * 2

    registry.register(OP, "triton", kernel)
    registry.register(OP, "eager", lambda x: x * 2)
    order = registry._STATE.priority["cpu"]
    registry.set_priority("cpu", ("triton", "eager"))
    yield registry.resolve(OP, "cpu", torch.float32), seen, fail
    registry.set_priority("cpu", order)
    registry._STATE.impls.pop(OP, None)
    status.clear_failures()
    tuner._TABLES.clear()


def _recorded(op) -> list[Call]:
    with recording.recording() as calls, torch.autocast("cpu", dtype=torch.bfloat16):
        op(torch.randn(8, 4))
    return calls


def check_autocast(op, seen) -> None:
    replay.replay(_recorded(op), [{}], backward=False)
    assert seen == [(True, torch.bfloat16)], seen


def test_replay_runs_under_the_recorded_autocast(probe):
    op, seen, _ = probe
    check_autocast(op, seen)


def test_negative_recording_without_autocast(probe, monkeypatch):
    op, seen, _ = probe
    monkeypatch.setattr(
        recording, "Call", lambda op, args, kwargs, cast: Call(op, args, kwargs)
    )
    must_fail(lambda: check_autocast(op, seen), "autocast state dropped")


def check_rebound(op, fail, model) -> None:
    """A backend that cannot run in the replay is fallen back before step 1: the
    recorded op and the model's op of the same query move to eager, the problem runs
    there, and the failure is recorded."""
    calls = _recorded(op)
    fail.append(BackendUnavailable(OP, {"cpu": "probe cannot run"}))
    with collecting(logging.WARNING) as logs:
        reports = replay.replay(calls, [{}], backward=False)
    assert reports[OP].failed == 0 and reports[OP].problems == 1, reports[OP]
    assert op.backend == "eager" and model.backend == "eager", (op.backend, model)
    q = op.query
    assert status.failed(OP, "triton", q.arch.name, q.dtype, q.static) is not None
    assert any(
        "backend triton failed, using eager" in r.getMessage() for r in logs.records
    )


def test_failed_replay_rebinds_the_ops(probe):
    op, _, fail = probe
    check_rebound(op, fail, registry.resolve(OP, "cpu", torch.float32))


def test_negative_replay_on_the_implementation(probe, monkeypatch):
    """A replay that calls the implementation directly leaves the ops bound."""
    op, _, fail = probe
    model = registry.resolve(OP, "cpu", torch.float32)
    monkeypatch.setattr(replay, "exercise", lambda o, run: run(o.fn))
    must_fail(lambda: check_rebound(op, fail, model), "replay on the implementation")


def check_oom_skipped(op, fail, err: BaseException) -> None:
    """A replay that runs out of memory is counted as skipped (out of memory) and the
    prewarm goes on with the op bound as it was."""
    calls = _recorded(op)
    fail.append(err)
    try:
        reports = replay.replay(calls, [{}], backward=False)
    except torch.OutOfMemoryError as oom:
        raise AssertionError("a replay OOM reached the caller") from oom
    assert reports[OP].oom == 1 and reports[OP].failed == 1, reports[OP]
    assert op.backend == "triton"


def test_replay_oom_is_skipped(probe):
    op, _, fail = probe
    check_oom_skipped(op, fail, torch.OutOfMemoryError("CUDA out of memory (test)"))


def test_negative_other_error_is_not_counted_as_oom(probe):
    op, _, fail = probe
    err = RuntimeError("an unrelated failure")
    must_fail(lambda: check_oom_skipped(op, fail, err), "not an OOM")


def check_longest_scaled(scale) -> None:
    """Only the int right after the boundary tensor (max_seqlen) follows the new
    longest segment; an unrelated int equal to it (a head count) is kept."""
    bounds = recording.snapshot(torch.tensor([0, 4, 8]))
    call = Call(None, (4, bounds, 4), {})
    args, _ = scale(call, replay.Target({8: 16}))
    assert args[1].tolist() == [0, 8, 16]
    assert args[2] == 8, "max_seqlen not scaled"
    assert args[0] == 4, "an unrelated int argument was rewritten"


def test_replay_scales_only_max_seqlen():
    check_longest_scaled(replay.scale)


def _every_equal_int(call, target):
    """The broken variant: every int equal to a recorded longest is replaced."""
    args, kwargs = replay.scale(call, target)
    return tuple(8 if type(a) is int and a == 4 else a for a in args), kwargs


def test_negative_every_equal_int_replaced():
    must_fail(lambda: check_longest_scaled(_every_equal_int), "heads rewritten")
