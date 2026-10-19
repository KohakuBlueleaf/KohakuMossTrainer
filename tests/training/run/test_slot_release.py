"""Objects that select a method once at build time are freed by reference counting
alone: the checkpoint manager, the progress bar and the staged sigma sampler are dead
(with the sampler's sigma table) right after `del`, with the cyclic collector off; with
the selected method bound to the object again each stays alive."""

import types
from functools import partial

import pytest
import torch

from kohakumoss.diffusion.timestep import MossTimestep
from kohakumoss.memory.timeline import NoTimeline
from kohakumoss.training.callbacks.progress import ProgressBar
from kohakumoss.training.checkpoint import CheckpointManager
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu


def _manager(root, strong: bool = False) -> tuple:
    mgr = CheckpointManager(root, 1, snapshot_ema=True)
    assert mgr._save is CheckpointManager._write
    if strong:
        mgr._save = types.MethodType(CheckpointManager._write, mgr)
    return (mgr,)


def _bar(strong: bool = False) -> tuple:
    bar = ProgressBar(4, "cpu", timeline=NoTimeline(), enabled=False)
    assert bar._vram_of is ProgressBar._vram
    if strong:
        bar._vram_of = types.MethodType(ProgressBar._vram, bar)
    return (bar,)


def _sampler(strong: bool = False) -> tuple:
    ts = MossTimestep(schedule="tech_report")
    sigma = ts.sample(8, torch.Generator().manual_seed(0), 0)
    assert sigma.shape == (8,) and sigma.dtype == torch.float32
    if strong:
        ts._sample = types.MethodType(MossTimestep._staged, ts)
    return ts, ts.stages[0].sigma


@pytest.mark.parametrize("build", ["manager", "bar", "sampler"])
def test_selected_slots_are_freed_on_del(build, tmp_path):
    make = {
        "manager": partial(_manager, tmp_path),
        "bar": _bar,
        "sampler": _sampler,
    }[build]
    check_released(make)
    must_fail(
        lambda: check_released(partial(make, strong=True)),
        f"the {build}'s selected method bound to it",
    )
