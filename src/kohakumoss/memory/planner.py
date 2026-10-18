"""`auto` memory profile: the fastest profile whose estimate fits the free memory."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from kohakumoss.memory.estimate import GIB, Estimate, Workload, estimate
from kohakumoss.memory.profiles import ORDER, resolve
from kohakumoss.offload.accel import device_budget
from kohakumoss.offload.encoders.text_memory import (
    TextSizes,
    choose_text_memory,
    text_footprint,
)
from kohakumoss.types.plans import MemoryPlan


@dataclass
class PlanReport:
    """Why `auto` chose what it chose: per-profile estimate and result."""

    budget: int
    chosen: str = ""
    rows: list[dict] = field(default_factory=list)
    forced: bool = False


PERSISTENT = (
    "context",
    "weights",
    "adapter",
    "optim",
    "optim_slots",
    "state_ring",
    "grad_ring",
    "ema",
)


def with_text(
    est: Estimate,
    sizes: TextSizes,
    budget: int,
    tokens: tuple[int, int],
    memory: str = "auto",
) -> tuple[int, str]:
    """(peak bytes of the step with the text encoder, its placement): resident text
    weights add to the DiT step peak; an offload / ring pass peaks on top of the
    persistent DiT parts only (its buffers are released before the step). `memory`
    'auto' picks the fastest placement that fits `budget`."""
    persistent = est.device(PERSISTENT)
    step = est.total
    if memory == "auto":
        memory, _ = choose_text_memory(
            sizes, budget - persistent, step - persistent, *tokens
        )
    fp = text_footprint(memory, sizes, *tokens)
    return max(step + fp["resident"], persistent + fp["pass"]), memory


def auto_plan(
    work: Workload,
    device: torch.device | str,
    overrides: dict | None = None,
    budget: int | None = None,
    trainable_base: bool = False,
    reject: Callable[[MemoryPlan], list[str]] | None = None,
    text: tuple[TextSizes, tuple[int, int], str | Callable] | None = None,
    force: bool = False,
) -> tuple[MemoryPlan, PlanReport]:
    """Walk profiles fastest-first; accept the first runnable one whose estimate fits
    `budget`. `reject(plan) -> problems` drops plans the other knobs cannot run. `text` =
    (text-encoder sizes, (tokens per pass, tokens per call), memory mode or
    `mode(plan)`) adds the online text encoder: each row records the placement it
    would use. A measured
    budget (`budget` None: free memory now) already excludes the context, so each
    estimate (context included) is compared with `room` = budget + its context part;
    a given `budget` is compared as is. When no runnable profile fits, raises
    NothingFits, unless `force`: then the last runnable one is chosen (marked
    `forced` in the report)."""
    measured = budget is None
    budget = device_budget(device) if measured else budget
    report = PlanReport(budget=budget)
    last: MemoryPlan | None = None
    for name in ORDER:
        try:
            plan = resolve(name, overrides, device, trainable_base)
        except ValueError as err:
            report.rows.append({"profile": name, "result": f"invalid: {err}"})
            continue
        problems = reject(plan) if reject is not None else []
        if problems:
            report.rows.append({"profile": name, "result": "; ".join(problems)})
            continue
        last = plan
        est: Estimate = estimate(work, plan)
        total = est.total
        row = {"profile": name, "estimate": est.total}
        context = est.parts.get("context", 0)
        room = budget + (context if measured else 0)
        if text is not None:
            mode = text[2](plan) if callable(text[2]) else text[2]
            total, row["text_memory"] = with_text(est, text[0], room, text[1], mode)
            row["estimate_with_text"] = total
        row["predicted"] = total
        row["room"] = room
        report.rows.append(row)
        if total > room:
            row["result"] = "over budget (estimate)"
            continue
        row["result"] = "chosen"
        report.chosen = name
        return plan, report
    if last is None:
        raise ValueError(f"auto memory plan: no profile is runnable: {report.rows}")
    if not force:
        raise NothingFits(report)
    report.chosen = f"{last.profile} (nothing fits, forced)"
    report.forced = True
    return last, report


class NothingFits(ValueError):
    """`auto` found no runnable profile whose estimate fits; `report` has every row."""

    def __init__(self, report: PlanReport):
        self.report = report
        rows = "; ".join(
            f"{r['profile']} {r['predicted'] / GIB:.2f} GiB"
            for r in report.rows
            if "predicted" in r
        )
        room = max((r["room"] for r in report.rows if "room" in r), default=0)
        super().__init__(
            f"MEMORY_PROFILE='auto': no profile fits {room / GIB:.2f} GiB ({rows}); "
            "set MEMORY_AUTO_FORCE=True to run the last one anyway, or pick a "
            "MEMORY_PROFILE"
        )
