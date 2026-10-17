"""A checkpoint is written once per step, into a `.partial` directory renamed when
complete; only complete, fully named directories are listed or resumed."""

import shutil

import pytest

from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail
from tests.training.test_resume import make_run

pytestmark = pytest.mark.cpu


def _count_writes(ckpt) -> list[int]:
    writes = []
    real = ckpt._save

    def spy(step, parts):
        writes.append(step)
        return real(step, parts)

    ckpt._save = spy
    return writes


def test_last_step_saved_once(tmp_path):
    model, drv, step, loader, bundle, ema, ckpt, parts = make_run(tmp_path, 6)
    writes = _count_writes(ckpt)
    drv.run(step, loader, bundle, ema, TrainState())
    assert writes == [3, 6]
    must_fail(lambda: _once([3, 6, 6]), "final step written twice")
    root = tmp_path / "checkpoints"
    assert sorted(p.name for p in root.iterdir()) == ["step_00000003", "step_00000006"]


def _once(writes: list[int]) -> None:
    assert len(writes) == len(set(writes)), writes


def test_partial_directories_are_never_resumed(tmp_path):
    model, drv, step, loader, bundle, ema, ckpt, parts = make_run(tmp_path, 3)
    drv.run(step, loader, bundle, ema, TrainState())
    done = ckpt.latest()
    torn = done.parent / "step_00000009.partial"
    shutil.copytree(done, torn)
    assert ckpt.latest() == done
    renamed = done.parent / "step_00000009"
    torn.rename(renamed)
    assert ckpt.latest() == renamed
