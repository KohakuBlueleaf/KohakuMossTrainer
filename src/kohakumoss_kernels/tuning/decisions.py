"""Kernel decision log: each tuned problem is recorded once, when its config settles
(`table`, `measured`, `inherited` or `default`), and flushed as one INFO table per
batch; after `go_live()` a measured decision is one INFO line. See
docs/internals/tuning.md#tune."""

from collections import Counter

from kohakumoss_common.log import get_logger, hl, log_table

SOURCES = ("table", "measured", "inherited", "default")
HEADERS = ("kernel", "from table", "measured", "inherited", "default", "chosen", "x")
log = get_logger(__name__)
_SEEN: set[tuple[str, str]] = set()
_BATCH: list[tuple[str, str, str, str]] = []
_STATE = {"live": False}


def label(config: dict) -> str:
    """A config for a log line: the value of a one-field config, else `k=v,...`."""
    if len(config) == 1:
        return str(next(iter(config.values())))
    return ",".join(f"{k}={v}" for k, v in sorted(config.items()))


def _times(timings: dict[str, float]) -> str:
    return ", ".join(
        f"{name.split('=', 1)[-1]} {ms:.3g} ms" for name, ms in timings.items()
    )


def undecided(kernel: str, key: str) -> None:
    """A binding that is not a decision (DEBUG only): provisional (a kernel-vs-compile
    problem queued for the retune) or untimed for the process (frozen, deferred, over
    the budget)."""
    log.debug(f"undecided {kernel} {key}")


def record(
    kernel: str, key: str, source: str, config: dict, timings: dict | None = None
) -> None:
    """One settled decision; a (kernel, key) already recorded is ignored."""
    if (kernel, key) in _SEEN:
        return
    _SEEN.add((kernel, key))
    chosen = label(config)
    detail = (
        _times(timings) if timings else ("stored" if source == "table" else "untimed")
    )
    log.debug(f"{kernel} {key}: {chosen} ({detail}), {source}")
    if not _STATE["live"]:
        _BATCH.append((kernel, key, source, chosen))
    elif source == "measured":
        log.info(f"{kernel} {hl(key)}: {hl(chosen)} ({detail}), measured")


def rows(batch: list[tuple[str, str, str, str]]) -> list[tuple]:
    """Summary rows per kernel: counts per source, then the chosen configs and how many
    problems took each (list cells, one table line per config, most common first)."""
    out = []
    for kernel in dict.fromkeys(k for k, *_ in batch):
        mine = [b for b in batch if b[0] == kernel]
        counts = Counter(source for _, _, source, _ in mine)
        chosen = Counter(c for *_, c in mine).most_common()
        configs, times = [c for c, _ in chosen], [n for _, n in chosen]
        out.append((kernel, *(counts.get(s, 0) for s in SOURCES), configs, times))
    return out


def flush(title: str) -> list[tuple]:
    """Log the decisions since the last flush as one table under `title`; returns its
    rows (empty: nothing logged)."""
    batch = list(_BATCH)
    _BATCH.clear()
    summary = rows(batch)
    if summary:
        log_table(log, f"{title}: {len(batch)} kernel decision(s)", HEADERS, summary)
    return summary


_LIVE_HOOKS: list = []


def on_live(fn) -> None:
    """Register `fn`, run once by `go_live` after the startup flush."""
    _LIVE_HOOKS.append(fn)


def go_live(title: str = "kernel decisions by the end of step 1") -> None:
    """Flush the startup batch; later measured decisions log one INFO line each; then
    the `on_live` hooks run."""
    flush(title)
    _STATE["live"] = True
    for fn in _LIVE_HOOKS:
        fn()


def clear() -> None:
    """Forget every recorded decision and leave live mode (tests)."""
    _SEEN.clear()
    _BATCH.clear()
    _STATE["live"] = False
