"""The end-of-run resume checkpoint follows FINAL_CHECKPOINT: "auto" writes it only when
periodic checkpoints are on, True / False force it (model-free: a recording manager)."""

import pytest

from kohakumoss.training.build.loop import final_checkpoint
from kohakumoss.training.checkpoint import CheckpointManager
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _k(final="auto", every=0, minutes=0) -> dict:
    return {"FINAL_CHECKPOINT": final, "SAVE_EVERY": every, "SAVE_MINUTES": minutes}


def check_rule(rule) -> None:
    assert rule(_k()) is False
    assert rule(_k(every=500)) is True
    assert rule(_k(minutes=30)) is True
    assert rule(_k(final=True)) is True
    assert rule(_k(final=False, every=500)) is False


def test_final_checkpoint_rule():
    check_rule(final_checkpoint)
    must_fail(lambda: check_rule(lambda k: True), "always saving at the end")


def _writes(final: bool, tmp_path, already: bool = False) -> list:
    written = []
    mgr = CheckpointManager(tmp_path, 0, final=final)
    mgr.save = lambda step, parts: written.append(step)
    if already:
        mgr.last_saved = 7
    mgr.finish(7, {})
    return written


def test_finish_writes_only_when_on(tmp_path):
    assert _writes(True, tmp_path) == [7]
    assert _writes(False, tmp_path) == []
    assert _writes(True, tmp_path, already=True) == []
    must_fail(lambda: _assert_none(_writes(True, tmp_path)), "an unconditional finish")


def _assert_none(written: list) -> None:
    assert written == [], written
