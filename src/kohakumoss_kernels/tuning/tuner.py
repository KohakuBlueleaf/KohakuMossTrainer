"""Pick one config per (kernel, problem key): stored winner, else time the shortlist.

`set_mode("table")` binds stored winners and times a missing key once on its first
eager call while the tuner is open (before `window.freeze`, or inside
`window.tuning_window` / `window.timing_here`); a frozen or deferring process binds its
first candidate untimed. `set_mode("off")` binds a stored winner when present, else the
first candidate, and never times. See docs/internals/tuning.md#tune.
"""

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from kohakumoss_common.log import get_logger
from kohakumoss_kernels.dist import COLLECTIVE_DEVICE
from kohakumoss_kernels.errors import (
    BUILD_ERRORS,
    ConfigsFailed,
    describe,
    is_toolchain_error,
)
from kohakumoss_kernels.hardware.current import (
    rng_kept,
    saved_tensor_hooks_active,
    with_index,
)
from kohakumoss_kernels.hardware.device import (
    cache_dir,
    planning_device,
    read_json,
    save_json,
    tracing,
)
from kohakumoss_kernels.hardware.memory import release, trim, working
from kohakumoss_kernels.hardware.toolchain import version
from kohakumoss_kernels.tuning import decisions, window
from kohakumoss_kernels.tuning.timing import best_ms, burst_ms, graph_ms
from kohakumoss_kernels.tuning.window import STATE as WINDOW

MODES = ("table", "off")
OOM_SKIPPED: tuple[type[BaseException], ...] = (torch.OutOfMemoryError,)
WORKING_FACTOR = 8.0
WORKING_FLOOR = 256 << 20
TIMINGS = {
    "graph": graph_ms,
    "events": lambda fn, device: best_ms(fn, device, 10),
    "burst": burst_ms,
}
TAGGED = (
    "kohakumoss",
    "lycoris-lora",
    "triton",
    "triton-windows",
    "triton-xpu",
    "tilelang",
)
_STATE = {"mode": "table"}
# (device, kernel, key) bound untimed and unstored: frozen, deferred, or over the budget
DEFERRED: set[tuple[str, str, str]] = set()
log = get_logger(__name__)
_RESETS: list[Callable[[], None]] = []
TABLE_TAG = hashlib.sha1(
    "|".join(f"{d}={version(d) or '-'}" for d in TAGGED).encode()
).hexdigest()[:10]


def set_mode(mode: str) -> None:
    """Apply the run's TUNE ("table" | "off") at the start of the run."""
    if mode not in MODES:
        raise ValueError(f"TUNE={mode!r} not in {MODES}")
    _STATE["mode"] = mode


def get_mode() -> str:
    return _STATE["mode"]


def on_reset(fn: Callable[[], None]) -> None:
    """Register a clear of a binding cache built from tuned choices (run by `reset`)."""
    _RESETS.append(fn)


def reset() -> None:
    """Forget every settled choice, every failed problem and every binding built from
    one."""
    _CHOSEN.clear()
    _FAILED.clear()
    _CAPS.clear()
    _rebind()


def _rebind() -> None:
    """Drop every binding built from a choice (callers look their choices up again)."""
    for fn in _RESETS:
        fn()


class TuneTable:
    """Winning configs of one kernel on one card, keyed by problem. A write merges the
    rows on disk first; threads sharing the table write one at a time; a write that
    fails keeps the rows in memory (logged at WARNING by `save_json`)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict] = read_json(path) or {}
        self.lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        row = self.rows.get(key)
        return None if row is None else row["config"]

    def _write(self, new: dict[str, dict]) -> None:
        with self.lock:
            self.rows = {**(read_json(self.path) or {}), **self.rows, **new}
            save_json(self.path, self.rows)

    def replace(self, rows: dict[str, dict]) -> None:
        """Make `rows` the whole table (memory and disk)."""
        with self.lock:
            self.rows = dict(rows)
            save_json(self.path, self.rows)

    def put(
        self,
        key: str,
        config: dict,
        timings: dict,
        failures: dict,
        working: dict | None = None,
    ) -> None:
        """Store the winner of `key` with every timing, failure and (when measured)
        each config's working memory in MiB."""
        row = {"config": config, "timings_ms": timings, "failures": failures}
        if working:
            row["working_mib"] = working
        self._write({key: row})

    def put_configs(self, configs: dict[str, dict]) -> None:
        """Store several `key -> config` rows (no timings) in one write."""
        new = {
            key: {"config": config, "timings_ms": {}, "failures": {}}
            for key, config in configs.items()
        }
        if new:
            self._write(new)


_TABLES: dict[tuple[str, str], TuneTable] = {}
_CHOSEN: dict[tuple, dict] = {}
_CAPS: dict[str, int] = {}  # device -> the process's tuning budget after the prewarm
_FAILED: dict[tuple, ConfigsFailed] = {}  # problems whose every config failed here
_CARDS: dict[str, str] = {}  # device -> card key of its tables
_NEW: list[tuple[str, str, str, dict]] = []  # (card, kernel, key, config) since `agree`
TIMED: dict[str, int] = {}  # kernel -> problems whose shortlist this process timed
TIMED_SECS: dict[str, float] = {}  # kernel -> wall seconds spent timing shortlists
_WATCH: list[set[tuple[str, str]]] = []


@contextmanager
def watching():
    """Collect the `(kernel, key)` of every problem with a choice to make that is looked
    up inside (`select` with more than one candidate, `note`); yields the set."""
    seen: set[tuple[str, str]] = set()
    _WATCH.append(seen)
    try:
        yield seen
    finally:
        _WATCH.pop()
        if _WATCH:
            _WATCH[-1].update(seen)


def note(kernel: str, key: str) -> None:
    """Mark a stored-table lookup made outside `select` (inside `watching()`)."""
    if _WATCH:
        _WATCH[-1].add((kernel, key))


def stored(kernel: str, device: torch.device | str, key: str) -> bool:
    """Whether this card's table of `kernel` holds a winner for `key`."""
    return table(kernel, device).get(key) is not None


def _path(dev_key: str, kernel: str) -> Path:
    return cache_dir() / "tuning" / dev_key / TABLE_TAG / f"{kernel}.json"


def _card_table(card: str, kernel: str) -> TuneTable:
    tab = _TABLES.get((card, kernel))
    if tab is None:
        tab = _TABLES.setdefault((card, kernel), TuneTable(_path(card, kernel)))
    return tab


def table(kernel: str, device: torch.device | str) -> TuneTable:
    """The process-shared table of `kernel` on `device`'s card (per package versions)."""
    card = planning_device(device).key
    _CARDS[str(device)] = card
    return _card_table(card, kernel)


def config_name(config: dict) -> str:
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
    lean: int = 0,
    factor: float | None = None,
    cap: int | None = None,
) -> dict:
    """The config for problem `key`: a stored winner that is still a candidate, else
    the fastest of the shortlist (`timing`: events | graph | burst) within the working-memory
    bound (`factor` x the leanest, default WORKING_FACTOR); the shortlist's config
    `lean` is timed first, the first candidate stays the default. A config raising one
    of `errors` or exceeding the bound is skipped; one running out of memory otherwise
    binds the best that ran for this process only; when every config failed,
    `ConfigsFailed` is raised (again on every later lookup in this process). `cap`
    bounds every config's working memory in bytes: a config past it raises `OverBound`
    and nothing is stored or settled. Where `timing_allowed` is false the first
    candidate is returned unsettled (settled under TUNE=off or for a single candidate);
    after `window.freeze` (outside a window) or under a deferring process group a
    missing key binds untimed (`_bind_untimed`); with no `cap` and a process budget
    (`set_cap`), a key over it binds its first candidate untimed (`_over_budget`). See
    docs/internals/tuning.md#select."""
    ident = (str(device), kernel, key)
    ranked = list(candidates()) if _WATCH else None
    if _WATCH and len(ranked) > 1:
        _WATCH[-1].add((kernel, key))
    chosen = _CHOSEN.get(ident)
    if chosen is not None:
        return chosen
    failed = _FAILED.get(ident)
    if failed is not None:
        raise ConfigsFailed(kernel, failed.errors, failed.persist, failed.problem)
    tab = table(kernel, device)
    ranked = list(candidates()) if ranked is None else ranked
    row = tab.get(key)
    if row is not None:
        match = [c for c in ranked if _canon(c) == _canon(row)]
        if match:
            _CHOSEN[ident] = match[0]
            decisions.record(kernel, key, "table", match[0])
            return match[0]
    if not timing_allowed(in_backward, autograd) or len(ranked) == 1:
        if get_mode() == "off" or len(ranked) == 1:
            _CHOSEN[ident] = ranked[0]
            decisions.record(kernel, key, "default", ranked[0])
        return ranked[0]
    if WINDOW["defer"] or WINDOW["frozen"]:
        return _bind_untimed(ident, device, key, ranked, launch, errors)
    measure = TIMINGS[timing]
    TIMED[kernel] = TIMED.get(kernel, 0) + 1
    began = time.perf_counter()
    order = lean_first(ranked[:shortlist], lean)
    budgeted = cap is None and str(device) in _CAPS
    try:
        with rng_kept([device]):
            return _measure(
                ident, device, key, tab, order, launch, measure, errors, factor,
                _CAPS[str(device)] if budgeted else cap, trims=not budgeted,
            )  # fmt: skip
    except OverBound:
        if not budgeted:
            raise
        return _over_budget(ident, key, ranked)
    finally:
        TIMED_SECS[kernel] = TIMED_SECS.get(kernel, 0.0) + time.perf_counter() - began


def set_cap(device: torch.device | str, nbytes: int | None) -> None:
    """From the end of the prewarm on: every `select` on `device` that passes no `cap`
    and is timed (inside a tuning window) times its configs within `nbytes` (the
    prewarm's tuning budget; None: no cap); a config that runs out of memory returns
    the free blocks (`release`); a key over it binds its first candidate, untimed."""
    if nbytes is None:
        _CAPS.pop(str(device), None)
    else:
        _CAPS[str(device)] = int(nbytes)


def clear_caps() -> None:
    """Drop every process tuning budget (`set_cap`)."""
    _CAPS.clear()


def cap_of(device: torch.device | str) -> int | None:
    """The process tuning budget of `device` in bytes (`set_cap`), None when unset."""
    return _CAPS.get(str(device))


def _over_budget(ident, key: str, ranked: list) -> dict:
    """A budgeted key whose timing would pass the process's tuning budget: its first
    candidate binds for the process, untimed and unstored (in DEFERRED, one INFO
    line)."""
    _CHOSEN[ident] = ranked[0]
    DEFERRED.add(ident)
    decisions.undecided(ident[1], key)
    log.info(
        f"{ident[1]} {key}: over the tuning budget: first candidate, untimed "
        f"({config_name(ranked[0])})"
    )
    return ranked[0]


def _bind_untimed(ident, device, key: str, ranked: list, launch, errors) -> dict:
    """A key met outside a tuning window (`window.freeze`; one WARNING line) or under a
    deferring process group (`defer_if_grouped`; one INFO line): the first of `ranked`
    that builds and runs once, untimed, binds for the process (unstored, in DEFERRED);
    a config raising one of `errors` is skipped; when every one does, `ConfigsFailed`
    (the registry falls back)."""
    kernel = ident[1]
    frozen = WINDOW["frozen"]
    failures, toolchain = {}, []
    for config in ranked:
        try:
            with torch.enable_grad(), rng_kept([device]):
                launch(config)()
        except errors as err:
            failures[config_name(config)] = describe(err)[:300]
            toolchain.append(is_toolchain_error(err))
            continue
        _CHOSEN[ident] = config
        DEFERRED.add(ident)
        decisions.undecided(kernel, key)
        why = "met outside a tuning window" if frozen else "deferred under DDP"
        (log.warning if frozen else log.info)(
            f"{kernel} {key}: {why}: first candidate, untimed ({config_name(config)})"
        )
        return config
    err = ConfigsFailed(kernel, failures, all(toolchain), f"{kernel}:{key}")
    _FAILED[ident] = err
    raise err


def lean_first(shortlist: list, lean: int) -> list:
    """`shortlist` in timing order: its config `lean` first, the rest in rank order."""
    if not 0 < lean < len(shortlist):
        return shortlist
    return [shortlist[lean], *shortlist[:lean], *shortlist[lean + 1 :]]


def working_bound(lean: int | None, factor: float | None = None) -> int | None:
    """The working memory a config may use: `factor` (default WORKING_FACTOR) x the
    leanest config timed so far for the same problem, at least WORKING_FLOOR (None: no
    config ran yet)."""
    if lean is None:
        return None
    factor = WORKING_FACTOR if factor is None else factor
    return max(int(factor * lean), WORKING_FLOOR)


class OverBound(RuntimeError):
    """A config of `kernel` problem `key` would pass the absolute working-memory cap
    (`select(cap=...)`); nothing was stored or settled."""

    def __init__(self, kernel: str, key: str, cap: int):
        super().__init__(f"{kernel} {key}: a config passes the {cap >> 20} MiB cap")
        self.kernel, self.key, self.cap = kernel, key, cap


def capped_bound(relative: int | None, cap: int | None) -> int | None:
    """The tighter of the relative bound and the absolute cap (None: neither)."""
    if cap is None:
        return relative
    return cap if relative is None else min(relative, cap)


def _measure(
    ident, device, key, tab, shortlist, launch, measure, errors, factor=None, cap=None,
    trims=True,
) -> dict:  # fmt: skip
    """Time `shortlist` under the working-memory bound, store and settle the winner
    (see `select`); `trims`: each bounded config starts from an empty allocator cache
    (False: each config starts from the current cache; an out-of-memory config is
    followed by `release`)."""
    kernel = ident[1]
    timings, failures, best = {}, {}, None
    toolchain, short, used = [], [], {}
    lean = None
    for config in shortlist:
        relative = working_bound(lean, factor)
        bound = capped_bound(relative, cap)
        if bound is not None and trims:
            # with trims: every bounded config starts from an empty allocator cache
            trim(device)
        try:
            with torch.enable_grad(), working(device, bound) as w:
                ms = measure(launch(config), device)
        except OOM_SKIPPED:
            release(device)
            if w.enforced and bound == cap:
                raise OverBound(kernel, key, cap) from None
            if w.enforced:
                failures[config_name(config)] = (
                    f"over the working-memory bound {bound >> 20} MiB"
                )
                continue
            short.append(config_name(config))
            continue
        except errors as err:
            failures[config_name(config)] = describe(err)[:300]
            toolchain.append(is_toolchain_error(err))
            continue
        timings[config_name(config)] = ms
        used[config_name(config)] = round(w.peak / 2**20, 1)
        lean = w.peak if lean is None else min(lean, w.peak)
        if best is None or ms < timings[config_name(best)]:
            best = config
    if short:
        return _settle_short(ident, kernel, key, best or shortlist[0], short)
    if best is None:
        err = ConfigsFailed(kernel, failures, all(toolchain), f"{kernel}:{key}")
        _FAILED[ident] = err
        raise err
    tab.put(key, best, timings, failures, used)
    _CHOSEN[ident] = best
    card = _CARDS.get(ident[0])
    if card is not None:
        _NEW.append((card, kernel, key, best))
    decisions.record(kernel, key, "measured", best, timings)
    return best


def _settle_short(ident, kernel: str, key: str, config: dict, short: list) -> dict:
    """A shortlist some of whose configs ran out of memory while timing: `config`
    binds for this process only (nothing stored, so a later process measures again)."""
    _CHOSEN[ident] = config
    log.warning(
        f"{kernel}: {len(short)} configs out of memory while timing {key}; "
        f"{config_name(config)} bound for this process, not stored"
    )
    return config


def row(kernel: str, device: torch.device | str, key: str) -> dict | None:
    """This card's stored row of problem `key` (config, timings, failures, the working
    memory per config when measured, the source key when inherited)."""
    return table(kernel, device).rows.get(key)


def inherit(kernel: str, device: torch.device, key: str, source: str) -> dict:
    """Settle and store problem `key` with the decision of problem `source` (settled or
    stored, on the same device), never timing it; the row names its source."""
    ident = (str(device), kernel, key)
    tab = table(kernel, device)
    config = _CHOSEN.get((str(device), kernel, source)) or tab.get(source)
    if config is None:
        raise KeyError(f"{kernel}: no decision for {source} to inherit")
    tab._write(
        {
            key: {
                "config": config,
                "timings_ms": {},
                "failures": {},
                "inherited": source,
            }
        }
    )
    _CHOSEN[ident] = config
    card = _CARDS.get(ident[0])
    if card is not None:
        _NEW.append((card, kernel, key, config))
    decisions.record(kernel, key, "inherited", config)
    return config


def defer_if_grouped() -> None:
    """At go-live (`decisions.go_live`): under a process group of two or more ranks,
    from then on no missing key is timed outside `window.timing_here()` (a
    `tuning_window` does not lift it): `select` binds its first candidate for the
    process, unstored, and names it in DEFERRED."""
    grouped = dist.is_available() and dist.is_initialized()
    window.defer(grouped and dist.get_world_size() > 1)


decisions.on_live(defer_if_grouped)


def timing_allowed(in_backward: bool = False, autograd: bool = False) -> bool:
    """Whether the calling context permits timing: not under `TUNE=off` or a trace, not
    inside an autograd backward unless `in_backward` (a backward kernel launching no
    autograd), and for `autograd` launches not under inference mode or saved-tensor
    hooks (a checkpointed forward); `select` also times nothing while `window` is
    frozen or deferring (`will_time`)."""
    if get_mode() == "off" or tracing():
        return False
    if autograd and (torch.is_inference_mode_enabled() or saved_tensor_hooks_active()):
        return False
    return in_backward or torch._C._current_graph_task_id() == -1


def will_time(in_backward: bool = False, autograd: bool = False) -> bool:
    """Whether `select` would time a missing key here: `timing_allowed` and the window
    neither frozen nor deferring (outside `timing_here` / `tuning_window`)."""
    if WINDOW["defer"] or WINDOW["frozen"]:
        return False
    return timing_allowed(in_backward, autograd)


def forget(kernel: str, device: torch.device, key: str) -> None:
    """Drop this process's choice for the problem (its config turned out unable to
    run), so the next `select` picks again from the remaining candidates."""
    _CHOSEN.pop((str(device), kernel, key), None)


def settled(kernel: str, device: torch.device, key: str) -> dict | None:
    """The config this process settled for the problem, else the stored winner, else
    None (nothing is timed)."""
    chosen = _CHOSEN.get((str(device), kernel, key))
    return chosen if chosen is not None else table(kernel, device).get(key)


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
        card, kernel = ident.split("|", 1)
        _card_table(card, kernel).replace(rows)
    _NEW.clear()
    reset()


def agreed(gathered: Sequence[Sequence[tuple]]) -> dict[tuple[str, str, str], dict]:
    """The config every rank binds per (card, kernel, key) measured since the last
    `agree`: the lowest rank's measurement (`gathered[r]`: rank r's rows)."""
    out: dict[tuple[str, str, str], dict] = {}
    for rows in gathered:
        for card, kernel, key, config in rows:
            out.setdefault((card, kernel, key), config)
    return out


def adopt(winners: dict[tuple[str, str, str], dict]) -> int:
    """Settle every winner on this process's devices of its card and store it; drops
    every binding when a winner replaces a different local choice. Returns how many
    winners differed from this process's own choice or stored row."""
    replaced, changed = False, 0
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for (card, kernel, key), config in winners.items():
        devices = [d for d, c in _CARDS.items() if c == card]
        if not devices:
            continue
        mine = [_CHOSEN.get((d, kernel, key)) for d in devices]
        differs = any(m is not None and _canon(m) != _canon(config) for m in mine)
        replaced = replaced or differs
        for d in devices:
            _CHOSEN[(d, kernel, key)] = config
        row = _card_table(card, kernel).get(key)
        new_row = row is None or _canon(row) != _canon(config)
        if new_row:
            rows.setdefault((card, kernel), {})[key] = config
        changed += new_row or differs
    for (card, kernel), configs in rows.items():
        _card_table(card, kernel).put_configs(configs)
    if replaced:
        _rebind()
    return changed


def positive_anywhere(count: int, group=None) -> bool:
    """Whether `count` is positive on any rank of `group` (one one-element all-reduce,
    on the backend's device)."""
    kind = COLLECTIVE_DEVICE.get(str(dist.get_backend(group)), "cpu")
    total = torch.tensor([count], device=with_index(kind))
    dist.all_reduce(total, group=group)
    return int(total.item()) > 0


def agree(group=None) -> int:
    """Between steps, on every rank: make every rank bind the same config for each
    problem some rank measured since the last call (the lowest rank's, `agreed`); a
    no-op without a process group of two or more ranks. Returns the winners that
    changed this rank's choices. See docs/internals/tuning.md#multi-process."""
    alone = not (dist.is_available() and dist.is_initialized())
    if alone or dist.get_world_size(group) < 2:
        _NEW.clear()
        return 0
    if not positive_anywhere(len(_NEW), group):
        return 0
    gathered: list = [None] * dist.get_world_size(group)
    mine = list(_NEW)
    _NEW.clear()
    dist.all_gather_object(gathered, mine, group=group)
    changed = adopt(agreed(gathered))
    if changed:
        log.info(f"tuning: {changed} choices agreed across ranks")
    return changed
