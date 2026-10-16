"""Pick one config per (kernel, problem key): stored winner, else time the shortlist.

Modes (`KOHAKUMOSS_TUNE` or `set_mode`): `auto` times a planner shortlist on the first
eager call of a new key and stores the winner; `off` takes the planner's first
candidate; `prewarm` is `auto` that callers drive ahead of training. Inside a trace
(torch.compile / graph capture) a missing key takes the first candidate untimed.
See docs/internals/tuning.md.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.distributed as dist

from kohakumoss_kernels.hardware.device import (
    cache_dir,
    planning_device,
    read_json,
    tracing,
    write_json,
)
from kohakumoss_kernels.tuning.timing import graph_ms

MODES = ("auto", "off", "prewarm")
_STATE = {"mode": os.environ.get("KOHAKUMOSS_TUNE", "auto")}


def set_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"tune mode {mode!r} not in {MODES}")
    _STATE["mode"] = mode


def get_mode() -> str:
    return _STATE["mode"]


class TuneTable:
    """Winning configs of one kernel on one card, keyed by problem."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict] = read_json(path) or {}

    def get(self, key: str) -> dict | None:
        row = self.rows.get(key)
        return None if row is None else row["config"]

    def put(self, key: str, config: dict, timings: dict, failures: dict) -> None:
        self.rows[key] = {"config": config, "timings_ms": timings, "failures": failures}
        write_json(self.path, self.rows)


_TABLES: dict[tuple[str, str], TuneTable] = {}
_CHOSEN: dict[tuple, dict] = {}


def table(device: torch.device | str, kernel: str) -> TuneTable:
    """The process-shared table of `kernel` on `device`'s card."""
    dev = planning_device(device)
    ident = (dev.key, kernel)
    if ident not in _TABLES:
        _TABLES[ident] = TuneTable(cache_dir() / "tuning" / dev.key / f"{kernel}.json")
    return _TABLES[ident]


def _name(config: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(config.items()))


def select(
    kernel: str,
    device: torch.device,
    key: str,
    candidates: Callable[[], Sequence[dict]],
    launch: Callable[[dict], Callable[[], object]],
    shortlist: int = 4,
    errors: tuple[type[BaseException], ...] = (RuntimeError,),
    in_backward: bool = False,
    autograd: bool = False,
) -> dict:
    """The config for problem `key`. `candidates()` is planner-ranked (best first);
    `launch(config)` returns a zero-argument call running that config on its own
    scratch / leaf copies. Timing runs under `enable_grad`; it is skipped (first
    candidate, not stored) while tracing, with `TUNE=off`, and inside an autograd
    backward (e.g. a checkpoint recompute) unless the caller is a backward kernel
    that launches no autograd (`in_backward=True`). A launch that runs autograd itself
    (`autograd=True`) is also not timed under inference mode or while saved-tensor
    hooks are active (a checkpointed forward); prewarm covers those."""
    ident = (str(device), kernel, key)
    chosen = _CHOSEN.get(ident)
    if chosen is not None:
        return chosen
    tab = table(device, kernel)
    stored = tab.get(key)
    if stored is not None:
        _CHOSEN[ident] = stored
        return stored
    ranked = list(candidates())
    if not timing_allowed(in_backward, autograd) or len(ranked) == 1:
        if get_mode() == "off" or len(ranked) == 1:
            _CHOSEN[ident] = ranked[0]
        return ranked[0]
    timings, failures, best = {}, {}, None
    for config in ranked[:shortlist]:
        try:
            with torch.enable_grad():
                ms = graph_ms(launch(config), device)
        except errors as err:
            failures[_name(config)] = f"{type(err).__name__}: {err}"[:300]
            continue
        timings[_name(config)] = ms
        if best is None or ms < timings[_name(best)]:
            best = config
    if best is None:
        raise RuntimeError(f"{kernel}: every shortlisted config failed: {failures}")
    tab.put(key, best, timings, failures)
    _CHOSEN[ident] = best
    return best


def timing_allowed(in_backward: bool = False, autograd: bool = False) -> bool:
    """Whether `select` may time candidates here (see `select`)."""
    if get_mode() == "off" or tracing():
        return False
    if autograd and (
        torch.is_inference_mode_enabled()
        or torch._C._autograd._top_saved_tensors_default_hooks(False) is not None
    ):
        return False
    return in_backward or torch._C._current_graph_task_id() == -1


def decided(kernel: str, device: torch.device, key: str) -> bool:
    """Whether `select` has a settled choice for this problem (callers bind only then;
    a provisional pick made while tracing or in a recompute is not settled)."""
    return (str(device), kernel, key) in _CHOSEN


def broadcast(src: int = 0, group=None) -> None:
    """Copy rank `src`'s tuning tables to every rank (memory and disk)."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    payload = [
        (
            {f"{k[0]}|{k[1]}": t.rows for k, t in _TABLES.items()}
            if dist.get_rank(group) == src
            else None
        )
    ]
    dist.broadcast_object_list(payload, src=src, group=group)
    for ident, rows in payload[0].items():
        dev_key, kernel = ident.split("|", 1)
        path = cache_dir() / "tuning" / dev_key / f"{kernel}.json"
        tab = _TABLES.setdefault((dev_key, kernel), TuneTable(path))
        tab.rows.update(rows)
        write_json(path, tab.rows)
    _CHOSEN.clear()
