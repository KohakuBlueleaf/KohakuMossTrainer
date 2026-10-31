"""CheckpointManager.save: a write that fails transiently is retried and the checkpoint
written; one that keeps failing skips the checkpoint (None, partial removed, the run goes
on) instead of raising (model-free: patched writers, one process)."""

import pytest

from kohakumoss.training.checkpoint import manager
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def _manager(monkeypatch, tmp_path, fails: int):
    calls = {"n": 0}

    def save_rng(path, rank, rng):
        calls["n"] += 1
        if calls["n"] <= fails:
            raise OSError("stale file handle")
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(manager, "save_rng", save_rng)
    monkeypatch.setattr(manager, "capture_rng", lambda *a: {})
    monkeypatch.setattr(manager, "SAVE_DEADLINE", 0.5)
    ckpt = manager.CheckpointManager(tmp_path, keep_last=2)
    final = tmp_path / "checkpoints" / "step_00000005"
    monkeypatch.setattr(ckpt, "_save", lambda mgr, step, parts: final)
    return ckpt, calls


def check_retried(monkeypatch, tmp_path, fails: int) -> None:
    ckpt, calls = _manager(monkeypatch, tmp_path, fails)
    assert ckpt.save(5, {"step_rng": None}) is not None
    assert ckpt.saved(5) and calls["n"] == fails + 1


def test_transient_write_failure_is_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, "_BACKOFF", 0.0)
    check_retried(monkeypatch, tmp_path, 1)


def test_persistent_failure_skips_the_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, "_BACKOFF", 0.0)
    ckpt, _ = _manager(monkeypatch, tmp_path, 10**6)
    assert ckpt.save(5, {"step_rng": None}) is None
    assert not ckpt.saved(5)
    assert not (tmp_path / "checkpoints" / "step_00000005.partial").exists()
    must_fail(
        lambda: check_retried(monkeypatch, tmp_path / "neg", 10**6), "never written"
    )
