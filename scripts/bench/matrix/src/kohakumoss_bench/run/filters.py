"""Host-side filters of a pool queue (the hosts file's `kinds`, `runs`, `priorities`):
each keeps the runs it selects plus every run a kept run needs (`Run.after`), in queue
order. See docs/performance/bench-matrix.md#hosts-file.
"""

from collections.abc import Callable
from fnmatch import fnmatch

from kohakumoss_bench.plan.config import Run


def closure(runs: list[Run], keep: set[str]) -> list[Run]:
    """`runs` restricted to `keep` and every run a kept run needs."""
    by_name = {r.name: r for r in runs}
    todo = list(keep)
    while todo:
        dep = by_name[todo.pop()].after
        if dep and dep not in keep:
            keep.add(dep)
            todo.append(dep)
    return [r for r in runs if r.name in keep]


def keep_kinds(runs: list[Run], kinds: list[str] | None) -> list[Run]:
    """Runs of the allowed kinds; None keeps all."""
    if kinds is None:
        return runs
    return closure(runs, {r.name for r in runs if r.kind in kinds})


def keep_names(runs: list[Run], patterns: list[str] | None) -> list[Run]:
    """Runs whose name matches one of `patterns` (fnmatch); None keeps all."""
    if patterns is None:
        return runs
    return closure(
        runs, {r.name for r in runs if any(fnmatch(r.name, p) for p in patterns)}
    )


def keep_priorities(
    runs: list[Run], ranks: list[int] | None, rank: Callable[[Run], int]
) -> list[Run]:
    """The queue's first run (it tunes) and the runs whose `rank` is in `ranks`; None
    keeps all."""
    if ranks is None or not runs:
        return runs
    keep = {runs[0].name} | {r.name for r in runs[1:] if rank(r) in ranks}
    return closure(runs, keep)
