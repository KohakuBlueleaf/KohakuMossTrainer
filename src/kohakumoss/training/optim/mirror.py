"""Schedules for optimisers that are not torch.optim.Optimizer subclasses (the offload
AdamWs): AnySchedule runs on a holder optimiser and its lr is mirrored onto the real
one after each step."""

import torch

from kohakumoss.training.optim.schedule import build_schedule


class _Holder(torch.optim.Optimizer):
    """Carries the schedule's lr; never steps."""

    def __init__(self, lr: float):
        super().__init__([torch.zeros(1)], {"lr": lr})

    def step(self, closure=None):
        return None


class MirrorSchedule:
    """`step()` advances the schedule and writes its lr into every group of `target`,
    each scaled by the group's initial lr ratio to group 0 (LoRA+ up-side groups keep
    their multiple)."""

    def __init__(self, target, schedule: dict, steps: int):
        self.target = target
        base = float(target.param_groups[0]["lr"])
        self.ratios = [float(g["lr"]) / base for g in target.param_groups]
        self.holder = _Holder(base)
        self.schedule = build_schedule(self.holder, schedule, steps)
        self._mirror()

    def _mirror(self) -> None:
        lr = self.holder.param_groups[0]["lr"]
        for group, ratio in zip(self.target.param_groups, self.ratios):
            group["lr"] = lr * ratio

    def step(self) -> None:
        self.schedule.step()
        self._mirror()

    def state_dict(self) -> dict:
        return self.schedule.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.schedule.load_state_dict(state)
        self._mirror()


def schedule_for(optimizer, schedule: dict, steps: int):
    """AnySchedule on a torch optimiser, MirrorSchedule on anything else."""
    if isinstance(optimizer, torch.optim.Optimizer):
        return build_schedule(optimizer, schedule, steps)
    return MirrorSchedule(optimizer, schedule, steps)
