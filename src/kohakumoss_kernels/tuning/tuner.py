"""Pick one config per (kernel, problem key): stored winner, else time the shortlist.

Modes (`KOHAKUMOSS_TUNE` or `set_mode`): `auto` times a planner shortlist on the first
eager call of a new key and stores the winner; `off` takes the planner's first
candidate; `prewarm` is `auto` that callers drive ahead of training. Inside a trace
(torch.compile / graph capture) a missing key takes the first candidate untimed.
See docs/internals/tuning.md.
"""

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.distributed as dist

from kohakumoss_kernels.errors import BUILD_ERRORS, ConfigsFailed, describe
from kohakumoss_kernels.hardware.device import (
    cache_dir,
    planning_device,
    read_json,
    tracing,
    write_json,
)
from kohakumoss_kernels.tuning.timing import best_ms, graph_ms

MODES = ("auto", "off", "prewarm")
TIMINGS = {"graph": graph_ms, "events": lambda fn, device: best_ms(fn, device, 10)}
TAGGED = ("kohakumoss", "triton", "triton-windows", "triton-xpu", "tilelang")
_STATE = {"mode": os.environ.get("KOHAKUMOSS_TUNE", "auto")}
_RESETS: list[Callable[[], None]] = []


def _version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "-"


TABLE_TAG = hashlib.sha1(
    "|".join(f"{d}={_version(d)}" for d in TAGGED).encode()
).hexdigest()[:10]


def set_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"tune mode {mode!r} not in {MODES}")
    _STATE["mode"] = mode


def get_mode() -> str:
    return _STATE["mode"]


def on_reset(fn: Callable[[], None]) -> None:
    """Register a clear of a binding cache built from tuned choices (run by `reset`)."""
    _RESETS.append(fn)


def reset() -> None:
    """Forget every settled choice and every binding built from one."""
    _CHOSEN.clear()
    for fn in _RESETS:
        fn()


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


def _path(dev_key: str, kernel: str) -> Path:
    return cache_dir() / "tuning" / dev_key / TABLE_TAG / f"{kernel}.json"


def table(device: torch.device | str, kernel: str) -> TuneTable:
    """The process-shared table of `kernel` on `device`'s card (per package versions)."""
    dev = planning_device(device)
    ident = (dev.key, kernel)
    if ident not in _TABLES:
        _TABLES[ident] = TuneTable(_path(dev.key, kernel))
    return _TABLES[ident]


def _name(config: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(config.items()))


def _canon(config: dict) -> str:
    """A config as stored on disk (tuples and lists compare equal)."""
    return json.dumps(config, sort_keys=True, default=str)


def select(
    kernel: str,
    device: torch.device,
    key: str,
    candidates: Callable[[], Sequence[dict]],
    launch: Callable[[dict], Callable[[], object]],
    shortlist: int = 4,
    errors: tuple[type[BaseException], ...] = BUILD_ERRORS,
    in_backward: bool = False,
    autograd: bool = False,
    timing: str = "events",
) -> dict:
    """The config for problem `key`: a stored winner that is still a candidate, else
    the fastest of the shortlist (`timing`: events | graph, see
    docs/internals/tuning.md). A config raising one of
    `errors` is skipped; out-of-memory always propagates. When timing is not allowed
    (see `timing_allowed`) the first candidate is returned unsettled."""
    ident = (str(device), kernel, key)
    chosen = _CHOSEN.get(ident)
    if chosen is not None:
        return chosen
    tab = table(device, kernel)
    ranked = list(candidates())
    stored = tab.get(key)
    if stored is not None:
        match = [c for c in ranked if _canon(c) == _canon(stored)]
        if match:
            _CHOSEN[ident] = match[0]
            return match[0]
    if not timing_allowed(in_backward, autograd) or len(ranked) == 1:
        if get_mode() == "off" or len(ranked) == 1:
            _CHOSEN[ident] = ranked[0]
        return ranked[0]
    measure = TIMINGS[timing]
    timings, failures, best = {}, {}, None
    for config in ranked[:shortlist]:
        try:
            with torch.enable_grad():
                ms = measure(launch(config), device)
        except torch.OutOfMemoryError:
            raise
        except errors as err:
            failures[_name(config)] = describe(err)[:300]
            continue
        timings[_name(config)] = ms
        if best is None or ms < timings[_name(best)]:
            best = config
    if best is None:
        raise ConfigsFailed(kernel, failures)
    tab.put(key, best, timings, failures)
    _CHOSEN[ident] = best
    return best


def timing_allowed(in_backward: bool = False, autograd: bool = False) -> bool:
    """Whether `select` may time candidates here: not under `TUNE=off` or a trace, not
    inside an autograd backward unless `in_backward` (a backward kernel launching no
    autograd), and for `autograd` launches not under inference mode or saved-tensor
    hooks (a checkpointed forward)."""
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
    """Copy rank `src`'s tuning tables to every rank (memory and disk), then `reset`
    every rank so live calls bind from the shared tables."""
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
        tab = _TABLES.setdefault((dev_key, kernel), TuneTable(_path(dev_key, kernel)))
        tab.rows = dict(rows)
        write_json(tab.path, tab.rows)
    reset()
