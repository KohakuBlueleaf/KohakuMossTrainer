"""AnySchedule wiring from the SCHEDULE knob."""

from copy import deepcopy

import torch
from anyschedule import AnySchedule
from torch.optim import Optimizer


def schedule_config(schedule: dict, steps: int) -> dict:
    """`{"mode": ..., ...}` (lr only) or `{"lr": {...}, ...}`; `end: -1` -> STEPS."""
    cfg = deepcopy(schedule)
    if "mode" in cfg:
        cfg = {"lr": cfg}
    for sub in cfg.values():
        sub.setdefault("value", 1.0)
        if sub.get("end", -1) == -1:
            sub["end"] = steps
        if sub["mode"] == "composer":
            first = sub.get("schedules", [{}])[0]
            first.setdefault("warmup", sub.pop("warmup", 0))
    return cfg


class Schedule(AnySchedule):
    """AnySchedule whose `load_state_dict` also writes the restored step's values (lr
    and every other scheduled key) into the optimiser's groups."""

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        apply_current(self)


def apply_current(schedule: AnySchedule) -> None:
    """The schedule's values at its current step written into its optimiser's groups."""
    groups = schedule.optimizer.param_groups
    for group, lr in zip(groups, schedule.get_last_lr()):
        group["lr"] = lr
    for key, factor in schedule.get_factors().items():
        if key == "lr":
            continue
        for group, base in zip(groups, schedule.base_param_groups[key]):
            if key not in group:
                continue
            if isinstance(group[key], torch.Tensor):
                group[key].fill_(factor * base)
            else:
                group[key] = factor * base


def build_schedule(optimizer: Optimizer, schedule: dict, steps: int) -> Schedule:
    return Schedule(optimizer, config=schedule_config(schedule, steps))
