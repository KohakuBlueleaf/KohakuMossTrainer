"""Prewarm problems already decided on this card: a problem (an op call at one planned
shape) maps to the `(kernel, key)` of every tuned choice it looked up when it last ran.
A problem whose every listed key has a stored winner is not run again; its live calls
bind from the tables. See docs/internals/tuning.md#prewarm.
"""

import hashlib
from contextlib import contextmanager

import torch

from kohakumoss_kernels.tuning import tuner

KERNEL = "prewarm_problems"


def problem_key(name: str, parts: tuple) -> str:
    """Table key of a problem: `name` plus a hash of its layout `parts` (a repr-stable
    tuple of shapes, dtypes, flags and small integer contents)."""
    return f"{name}|{hashlib.sha1(repr(parts).encode()).hexdigest()[:16]}"


class Manifest:
    """The decided problems of one device; `commit()` stores the problems run since."""

    def __init__(self, device: torch.device | str):
        self.device = device
        self.table = tuner.table(KERNEL, device)
        self.ran: dict[str, set[tuple[str, str]]] = {}

    def covered(self, key: str) -> bool:
        """Whether problem `key` ran before and every choice it made is stored."""
        row = self.table.get(key)
        return row is not None and all(
            tuner.stored(kernel, self.device, k) for kernel, k in row["selects"]
        )

    @contextmanager
    def running(self, key: str):
        """Run problem `key` inside: the choices it looks up are noted for `commit`."""
        with tuner.watching() as seen:
            yield
        self.ran[key] = seen

    def working(self, key: str) -> int | None:
        """The working bytes stored for prewarm `key` (None: none stored)."""
        row = self.table.get(key)
        return None if row is None else row.get("working")

    def keep_working(self, key: str, nbytes: int) -> None:
        """Store prewarm `key`'s working bytes (its device peak above the bytes
        allocated when it started)."""
        self.table.put_configs({key: {"working": int(nbytes)}})

    def commit(self) -> int:
        """Store every problem run whose choices are all decided; returns how many."""
        done = {
            key: {"selects": sorted([kernel, k] for kernel, k in seen)}
            for key, seen in self.ran.items()
            if all(tuner.stored(kernel, self.device, k) for kernel, k in seen)
        }
        self.table.put_configs(done)
        self.ran.clear()
        return len(done)
