"""Greedy consecutive runs of a sequence under a size limit."""

from collections.abc import Callable, Iterable


def runs_within(items: Iterable, size_of: Callable, limit: int | None) -> list[list]:
    """Consecutive runs of `items` whose summed `size_of(item)` stays within `limit`
    (an item larger than `limit` is a run of its own; all items one run when None)."""
    items = list(items)
    if limit is None:
        return [items]
    out, current, size = [], [], 0
    for item in items:
        n = size_of(item)
        if current and size + n > limit:
            out.append(current)
            current, size = [], 0
        current.append(item)
        size += n
    return out + ([current] if current else [])
