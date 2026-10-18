"""Resume restores the scheduled lr into every optimiser the schedule drives: the lr
holder of the per-parameter fused step and the mirrored offload AdamWs run their first
resumed update at the checkpoint's lr, not the schedule's step-0 lr (model-free:
explicit parameter lists)."""

import pytest
import torch
from anyschedule import AnySchedule

from kohakumoss.offload.optim.gpu_adamw import MasterAdamW
from kohakumoss.precision import build_precision
from kohakumoss.training.optim import schedule as schedule_module
from kohakumoss.training.optim.bundle.build import build_bundle
from kohakumoss.training.optim.bundle.mirror import MirrorSchedule
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SCHED = {"mode": "cosine", "warmup": 2, "end": 10}
STEPS = 10
LR = 1e-2


def _param() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.ones(4, 3))


def _mirror_lr_after_resume(walked: int) -> tuple[float, float]:
    """(lr the source schedule holds after `walked` steps, lr the resumed one sets)."""
    first = MirrorSchedule(MasterAdamW([_param()], lr=LR), SCHED, STEPS)
    for _ in range(walked):
        first.step()
    target = MasterAdamW([_param()], lr=LR)
    resumed = MirrorSchedule(target, SCHED, STEPS)
    resumed.load_state_dict(first.state_dict())
    return first.target.param_groups[0]["lr"], target.param_groups[0]["lr"]


def _fused_lr_after_resume(walked: int) -> tuple[float, float]:
    """The same through the fused per-parameter bundle's lr holder."""

    def bundle():
        p = _param()
        groups = [{"params": [p], "lr": LR, "names": ["p"], "weight_decay": 0.0}]
        return build_bundle(
            groups,
            torch.device("cpu"),
            "adamw",
            {"lr": LR, "betas": (0.9, 0.99), "eps": 1e-8},
            SCHED,
            STEPS,
            build_precision("fp32", ()),
            0.0,
            fused_step=True,
        )

    first = bundle()
    for _ in range(walked):
        first.lr_source._opt_called = True
        first.schedule.step()
    resumed = bundle()
    resumed.load_state_dict(first.state_dict())
    return first.lr, resumed.lr


def _same(pair: tuple[float, float]) -> None:
    assert pair[0] == pytest.approx(pair[1], rel=1e-9), pair


@pytest.mark.parametrize("resume", [_mirror_lr_after_resume, _fused_lr_after_resume])
def test_resume_writes_the_scheduled_lr(resume, monkeypatch):
    walked = 6
    got = resume(walked)
    _same(got)
    assert got[1] != pytest.approx(resume(0)[1]), "lr did not move with the schedule"
    monkeypatch.setattr(
        schedule_module.Schedule, "load_state_dict", AnySchedule.load_state_dict
    )
    must_fail(lambda: _same(resume(walked)), "the scheduler state alone restored")
