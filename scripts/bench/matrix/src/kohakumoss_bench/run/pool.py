"""A card pool: the cards of one model on a host share one queue of units
(`matrix.units`). The first unit (the tuning run) runs alone; the pool's other cards
start once it has finished. The final unit (the restart run) is taken when every other
unit has finished. See docs/performance/bench-matrix.md#host-runner.
"""

import threading
import time
from collections import deque

from kohakumoss_bench.plan.config import Run

POLL_S = 5.0


class Pool:
    """Units of one card model, handed to the pool's card workers."""

    def __init__(self, model: str, units: list[list[Run]], final: list[Run], stop):
        self.model = model
        self.pending = deque(units[1:])
        self.first = units[0] if units else None
        self.final = final or None
        self.stop = stop
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.running = 0
        if self.first is None:
            self.gate.set()

    def total(self) -> int:
        return (
            (len(self.first) if self.first else 0)
            + sum(len(u) for u in self.pending)
            + (len(self.final) if self.final else 0)
        )

    def take(self) -> tuple[list[Run], bool] | None:
        """(unit, whether it is the first) for a free card; blocks while the first or
        the final unit has to wait; None when nothing is left or the host stopped."""
        while not self.stop.is_set():
            with self.lock:
                if self.first is not None:
                    unit, self.first = self.first, None
                    self.running += 1
                    return unit, True
                if self.gate.is_set():
                    if self.pending:
                        self.running += 1
                        return self.pending.popleft(), False
                    if self.final is not None and self.running == 0:
                        unit, self.final = self.final, None
                        self.running += 1
                        return unit, False
                    if self.final is None:
                        return None
            if self.gate.is_set():
                time.sleep(POLL_S)
            else:
                self.gate.wait(POLL_S)
        return None

    def done(self, first: bool) -> None:
        with self.lock:
            self.running -= 1
        if first:
            self.gate.set()
