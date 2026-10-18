"""A sticky device error met while replaying a recorded problem stops the prewarm (the
context is unusable); any other failure of a problem is skipped and counted."""

import pytest
import torch

from kohakumoss_kernels import recording, registry, status
from kohakumoss_kernels.tuning import replay, tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OP = "test_replay_sticky"
STICKY = "CUDA error: an illegal memory access was encountered"


@pytest.fixture
def sticky_op(tmp_path, monkeypatch):
    """`OP` (eager only), recorded once; it raises a sticky error when `armed`."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(status, "_CHECKS", {})
    tuner._TABLES.clear()
    tuner.reset()
    armed = {"on": False}

    def double(x):
        if armed["on"]:
            raise RuntimeError(STICKY)
        return x * 2

    registry.register(OP, "eager", double)
    order = registry._STATE.priority["cpu"]
    op = registry.resolve(OP, "cpu", torch.float32)
    with recording.recording() as calls:
        op(torch.zeros(8))
    armed["on"] = True
    yield calls
    registry.set_priority("cpu", order)
    registry._STATE.impls.pop(OP, None)
    tuner._TABLES.clear()
    tuner.reset()


def check_sticky_stops(calls) -> None:
    try:
        replay.replay(calls, [{}], backward=False)
    except RuntimeError as err:
        assert "illegal memory access" in str(err), err
        return
    raise AssertionError("the replay carried on past a sticky error")


def test_sticky_error_stops_the_replay(sticky_op):
    check_sticky_stops(sticky_op)


def test_negative_sticky_error_skipped(sticky_op, monkeypatch):
    monkeypatch.setattr(replay, "is_sticky_error", lambda err, device=None: False)
    must_fail(lambda: check_sticky_stops(sticky_op), "replay carried on")
