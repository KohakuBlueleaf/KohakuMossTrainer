"""`auto` memory profile: the fastest profile whose estimate (probe-calibrated) fits."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from kohakumoss.training.memory.estimate import Estimate, Workload, estimate
from kohakumoss.training.memory.profiles import ORDER, resolve
from kohakumoss.types.plans import MemoryPlan

HEADROOM = 0.92


@dataclass
class PlanReport:
    """Why `auto` chose what it chose: per-profile estimate, probe and verdict."""

    budget: int
    chosen: str = ""
    rows: list[dict] = field(default_factory=list)


def device_budget(device: torch.device | str) -> int:
    """Bytes `auto` may plan for on `device` (free memory now, with headroom)."""
    device = torch.device(device)
    match device.type:
        case "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            return int(free * HEADROOM)
        case "xpu":
            props = torch.xpu.get_device_properties(device)
            return int(props.total_memory * HEADROOM)
        case _:
            return 1 << 62


def auto_plan(
    work: Workload,
    device: torch.device | str,
    overrides: dict | None = None,
    budget: int | None = None,
    probe: Callable[[MemoryPlan], int] | None = None,
    trainable_base: bool = False,
    reject: Callable[[MemoryPlan], list[str]] | None = None,
) -> tuple[MemoryPlan, PlanReport]:
    """Walk profiles fastest-first; accept the first runnable one whose estimate fits
    `budget`. `reject(plan) -> problems` drops plans the other knobs cannot run;
    `probe(plan) -> measured peak bytes` runs one real step, its ratio to the estimate
    rescales later estimates, and a probe over budget rejects the plan."""
    budget = device_budget(device) if budget is None else budget
    report = PlanReport(budget=budget)
    ratio = 1.0
    last: MemoryPlan | None = None
    for name in ORDER:
        try:
            plan = resolve(name, overrides, device, trainable_base)
        except ValueError as err:
            report.rows.append({"profile": name, "verdict": f"invalid: {err}"})
            continue
        problems = reject(plan) if reject is not None else []
        if problems:
            report.rows.append({"profile": name, "verdict": "; ".join(problems)})
            continue
        last = plan
        est: Estimate = estimate(work, plan)
        predicted = int(est.total * ratio)
        row = {"profile": name, "estimate": est.total, "predicted": predicted}
        report.rows.append(row)
        if predicted > budget:
            row["verdict"] = "over budget (estimate)"
            continue
        if probe is not None:
            measured = int(probe(plan))
            row["measured"] = measured
            ratio = measured / max(est.total, 1)
            if measured > budget:
                row["verdict"] = "over budget (probe)"
                continue
        row["verdict"] = "chosen"
        report.chosen = name
        return plan, report
    if last is None:
        raise ValueError(f"auto memory plan: no profile is runnable: {report.rows}")
    report.chosen = f"{last.profile} (nothing fits)"
    return last, report
