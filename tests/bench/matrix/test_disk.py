"""Bench disk use: a row's checkpoints and exports go once no queued row reads them
(results, metrics and logs stay), and a row waits while the runs disk is below its floor.
"""

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.plan.config import Card, Options, Run
from kohakumoss_bench.plan.knobs import EXPORT, GRID, TRAIN
from kohakumoss_bench.run import disk
from kohakumoss_bench.run.host import HostRunner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
KEEP = ("result.json", "log.txt", "train/metrics.jsonl", "trace_step.json.gz")


def _queue() -> list[Run]:
    ckpt = "{run:quality}/train/checkpoints/step_00000100/trainable.safetensors"
    return [
        Run("speed", "speed", TRAIN, {"OUTPUT_DIR": "{self}"}, 1),
        Run("quality", "quality", TRAIN, {"OUTPUT_DIR": "{self}"}, 1),
        Run(
            "export",
            "export",
            EXPORT,
            {"ADAPTER": ckpt, "OUT": "{self}/a.safetensors"},
            1,
        ),
        Run(
            "grid",
            "grid",
            GRID,
            {"CHECKPOINTS": [("{run:export}/a.safetensors", 1.0)]},
            1,
        ),
    ]


def _fill(root: Path, name: str) -> Path:
    d = root / "card" / name
    for rel in (*KEEP, "train/checkpoints/step_00000100/trainable.safetensors"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(b"x" * 1024)
    (d / "train" / "export").mkdir(parents=True, exist_ok=True)
    (d / "train" / "export" / "lycoris.safetensors").write_bytes(b"x" * 1024)
    (d / "a.safetensors").write_bytes(b"x" * 1024)
    return d


def _artifacts(d: Path) -> bool:
    return bool(disk.artifacts(d))


def check_cleanup(cls, root: Path) -> None:
    dirs = {r.name: _fill(root, r.name) for r in _queue()}
    outside = _fill(root.parent / "elsewhere", "speed")
    c = cls(_queue(), root)
    c.finished("speed", dirs["speed"])
    assert not _artifacts(dirs["speed"]), "an unread row kept its checkpoints"
    c.finished("quality", dirs["quality"])
    assert _artifacts(dirs["quality"]), "checkpoints deleted while an export reads them"
    c.finished("export", dirs["export"])
    assert not _artifacts(dirs["quality"]), "checkpoints kept after their last reader"
    assert _artifacts(dirs["export"]), "an export deleted while a grid reads it"
    c.finished("grid", dirs["grid"])
    assert not _artifacts(dirs["export"]), "an export kept after its last reader"
    for d in dirs.values():
        assert all((d / rel).is_file() for rel in KEEP), f"{d}: a kept file was deleted"
    c.finished("outside", outside)
    assert _artifacts(outside), "a directory outside the runs root was cleaned"
    assert c.freed_gib > 0


class _Eager(disk.Cleanup):
    """Broken: deletes a row's artifacts as soon as it finishes."""

    def finished(self, name, run_dir):
        self.readers.clear()
        return super().finished(name, run_dir)


class _Never(disk.Cleanup):
    """Broken: never deletes."""

    def _remove(self, run_dir):
        return 0.0


def test_cleanup_after_the_last_reader(tmp_path):
    check_cleanup(disk.Cleanup, tmp_path / "runs")


def test_cleanup_check_catches_eager_and_missing_cleanup(tmp_path):
    must_fail(lambda: check_cleanup(_Eager, tmp_path / "a"), "deletes before readers")
    must_fail(lambda: check_cleanup(_Never, tmp_path / "b"), "never deletes")


def check_sweep(sweep, root: Path) -> None:
    dirs = {r.name: _fill(root, r.name) for r in _queue()}
    previous = {"speed", "quality"}
    cleanup = disk.Cleanup(_queue(), root)
    fake = SimpleNamespace(
        finished=lambda name, model: (
            {"dir": str(dirs[name])} if name in previous else None
        ),
        _finished=lambda model, name, d: cleanup.finished(name, d),
    )
    sweep(fake, "m", _queue())
    assert not _artifacts(dirs["speed"]), "a resumed unread row kept its checkpoints"
    assert _artifacts(dirs["quality"]), "a resumed row's checkpoints went before reads"
    assert cleanup.done == previous


def test_start_sweeps_resumed_rows(tmp_path):
    check_sweep(HostRunner.sweep, tmp_path / "runs")


def test_sweep_check_catches_a_missing_sweep(tmp_path):
    must_fail(lambda: check_sweep(lambda *a: None, tmp_path / "runs"), "no sweep")


def check_retire(retire, root: Path) -> None:
    d = _fill(root, "rerun")
    old = (d / "train" / "metrics.jsonl").read_bytes()
    retire(d)
    assert not (d / "train").exists(), "a rerun would append to the old metric rows"
    moved = d / "train.attempt-1"
    assert (moved / "metrics.jsonl").read_bytes() == old, "the old attempt was lost"
    assert not (moved / "checkpoints").exists(), "the old attempt kept its checkpoints"
    (d / "train").mkdir()
    retire(d)
    assert (d / "train.attempt-2").is_dir() and (d / "train.attempt-1").is_dir()
    assert disk.retire(root / "never-ran") is None


def test_rerun_starts_with_empty_metrics(tmp_path):
    check_retire(disk.retire, tmp_path / "runs")


def test_retire_check_catches_an_append(tmp_path):
    must_fail(lambda: check_retire(lambda d: None, tmp_path / "runs"), "no retire")


def _holder(tmp_path, frees: list[float]) -> SimpleNamespace:
    statuses: list[dict] = []
    fake = SimpleNamespace(
        opts=Options(disk_poll_s=0.0),
        stop=threading.Event(),
        out=tmp_path,
        statuses=statuses,
        _status=lambda card, **f: statuses.append(f),
    )
    it = iter(frees)
    fake.free = lambda path: next(it)
    return fake


def check_hold(monkeypatch, tmp_path) -> None:
    fake = _holder(tmp_path, [10.0, 20.0, 49.0, 200.0])
    monkeypatch.setattr(disk, "free_gib", fake.free)
    run = Run("r", "speed", TRAIN, {"ALGO": "lora"}, 1)
    assert HostRunner.wait_disk(fake, Card(0, "c", 24.0, 86), run)
    held = [s for s in fake.statuses if s.get("held")]
    assert len(held) == 1, f"the hold was reported {len(held)} times"
    assert fake.statuses[-1] == {"held": None}, "the hold was not cleared"


def test_row_waits_below_the_floor_and_reports_once(monkeypatch, tmp_path):
    check_hold(monkeypatch, tmp_path)
    opts = Options()
    full = Run("f", "coverage", TRAIN, {"ALGO": "full"}, 1)
    assert disk.floor_gib(full, opts) == 2 * opts.full_checkpoint_gib
    assert disk.floor_gib(Run("l", "speed", TRAIN, {}, 1), opts) == opts.disk_floor_gib


def test_hold_check_catches_a_missing_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(disk, "floor_gib", lambda run, opts: 0.0)
    must_fail(lambda: check_hold(monkeypatch, tmp_path), "no floor")
