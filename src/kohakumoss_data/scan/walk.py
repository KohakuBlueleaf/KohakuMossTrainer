"""Walk items: what a scanning source lists cheaply (one os.scandir per directory) and
what it probes expensively. An item has a key (stable across rescans), an 8-int stamp
(stats of the files its rows come from; -1 = missing) and a picklable probe."""

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from kohakumoss_data.types import CatalogRow

SIDECARS = (".txt", ".caption", ".json")
MISSING = (-1, -1)


@dataclass(frozen=True)
class WalkItem:
    """`probe()` returns the item's rows (one for a file, many for a container)."""

    key: str
    stamp: tuple
    probe: Callable[[], list[CatalogRow]]


@dataclass
class Listing:
    """One directory: sorted entry names, which are directories (following links) and
    symlinks, and every file's (size, mtime_ns) by `os.path.normcase` name."""

    names: list[str] = field(default_factory=list)
    dirs: set[str] = field(default_factory=set)
    links: set[str] = field(default_factory=set)
    stats: dict[str, tuple[int, int]] = field(default_factory=dict)

    def stat(self, name: str) -> tuple[int, int]:
        return self.stats.get(os.path.normcase(name), MISSING)


def listing(path: str) -> Listing:
    """`path` listed once; an absent directory lists empty."""
    out = Listing()
    if not os.path.isdir(path):
        return out
    with os.scandir(path) as entries:
        for entry in entries:
            out.names.append(entry.name)
            try:
                if entry.is_symlink():
                    out.links.add(entry.name)
                if entry.is_dir():
                    out.dirs.add(entry.name)
                    continue
                st = entry.stat()
            except OSError:
                continue
            out.stats[os.path.normcase(entry.name)] = (st.st_size, st.st_mtime_ns)
    out.names.sort()
    return out


def file_stamp(entries: Listing, name: str) -> tuple:
    """(size, mtime_ns) of `name` and of its .txt / .caption / .json sidecars."""
    stem = os.path.splitext(name)[0]
    stamp = entries.stat(name)
    for ext in SIDECARS:
        stamp += entries.stat(stem + ext)
    return stamp


class Listings:
    """Directory listings made on first use (manifest sources: one per image folder)."""

    def __init__(self):
        self._by_dir: dict[str, Listing] = {}

    def stat(self, path: str) -> tuple[int, int]:
        folder, name = os.path.split(os.path.abspath(path))
        entries = self._by_dir.get(folder)
        if entries is None:
            entries = self._by_dir[folder] = listing(folder)
        return entries.stat(name)


def rows_of(items: list[WalkItem]) -> Iterator[CatalogRow]:
    """Every item's rows in walk order (the uncached scan)."""
    for item in items:
        yield from item.probe()
