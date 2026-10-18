"""The pipeline's final export, model-free on two gloo ranks: before the export rank 0's
live trainable set holds every rank's trained stage entries (each rank trained its own
key), not only its own."""

import json

import pytest
import torch

from kohakumoss.parallel.telemetry.selftest import spawn_results
from kohakumoss.training.drivers.pipeline import PipelineParts
from kohakumoss.training.ema.averagers import NoEMA
from kohakumoss.types.plans import ParallelPlan
from tests._util.negative import must_fail

NAMES = ("stage0.w", "stage1.w")


class NamedState:
    """A trainable set as the checkpoint parts see it: named tensors in and out."""

    def __init__(self, values: dict[str, torch.Tensor]):
        self.values = values

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.values.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.values = {k: v.clone() for k, v in state.items()}


def export_worker(env, path: str, merge: bool) -> None:
    """Rank r trained NAMES[r] to r + 1; rank 0 reports its live values."""
    trainable = NamedState({n: torch.zeros(2) for n in NAMES})
    mine = NAMES[env.rank]
    trainable.values[mine] = torch.full((2,), float(env.rank + 1))
    parts = PipelineParts(
        trainable,
        NoEMA(),
        None,
        None,
        None,
        local_names={mine},
        rank=env.rank,
        world=env.world,
    )
    if merge:
        parts.before_export()
    values = {k: v.tolist() for k, v in trainable.values.items()}
    with open(f"{path}.{env.rank}.json", "w", encoding="utf-8") as f:
        json.dump(values, f)


def _run(merge: bool) -> dict:
    plan = ParallelPlan(kind="test", backend="gloo", devices=(0, 1))
    return spawn_results(export_worker, plan, merge, device_type="cpu")[0]


def check_merged(rank0: dict) -> None:
    assert rank0[NAMES[0]] == [1.0, 1.0], rank0
    assert rank0[NAMES[1]] == [2.0, 2.0], "rank 1's stage missing on rank 0"


@pytest.mark.cpu
@pytest.mark.budget(60)  # spawns two gloo rank processes twice
def test_export_holds_every_stage():
    check_merged(_run(True))
    must_fail(lambda: check_merged(_run(False)), "rank 0's own stages only")
