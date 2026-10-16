"""AnySchedule wiring from the SCHEDULE knob."""

from copy import deepcopy

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


def build_schedule(optimizer: Optimizer, schedule: dict, steps: int) -> AnySchedule:
    return AnySchedule(optimizer, config=schedule_config(schedule, steps))
