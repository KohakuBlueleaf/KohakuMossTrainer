"""CallbackRunner: the per-step action list, built once from each callback's cadence.

Callbacks with `every == 1` run every step; periodic ones run when the step reaches the
precomputed next firing step (one integer compare per step for all of them). Firing
order is the order the callbacks were given. Callbacks themselves never check cadence.
"""

import math
import time
from collections.abc import Callable

import torch.distributed as dist

from kohakumoss.training.state import TrainState


class CallbackRunner:
    """`run(state, out)` once per optimiser step, after `state.step` advanced."""

    def __init__(self, callbacks: list, start_step: int = 0):
        self.ordered = [cb for cb in callbacks if cb.every > 0]
        self.always = [cb for cb in self.ordered if cb.every == 1]
        self.periods = sorted({cb.every for cb in self.ordered if cb.every > 1})
        self.reset(start_step)

    def reset(self, step: int) -> None:
        """Recompute the next firing step after `step` (resume)."""
        self.next_fire = self._next_after(step)

    def _next_after(self, step: int) -> float:
        return (
            min(((step // p) + 1) * p for p in self.periods)
            if self.periods
            else math.inf
        )

    def due(self, step: int) -> list:
        """Callbacks firing at `step` (every-step ones included), in given order."""
        return [cb for cb in self.ordered if step % cb.every == 0]

    def run(self, state: TrainState, out) -> None:
        if state.step == self.next_fire:
            for cb in self.due(state.step):
                cb.on_step_end(state, out)
            self.next_fire = self._next_after(state.step)
            return
        for cb in self.always:
            cb.on_step_end(state, out)


class SaveCheckpoint:
    """Every `every` steps: write a checkpoint from the live parts."""

    def __init__(self, every: int, checkpoints, parts):
        self.every = every
        self.checkpoints = checkpoints
        self.parts = parts

    def on_step_end(self, state: TrainState, out) -> None:
        self.checkpoints.save(state.step, self.parts(state))


def _own_decision(due: bool) -> bool:
    return due


def rank0_decision(due: bool) -> bool:
    """Rank 0's decision on every rank (one object broadcast)."""
    box = [due]
    dist.broadcast_object_list(box, src=0)
    return bool(box[0])


class TimedCheckpoint:
    """Every `every` steps: write a checkpoint when `minutes` of wall time passed since
    the last. `agree` makes every rank take rank 0's decision (the save is collective).
    """

    def __init__(
        self,
        minutes: float,
        checkpoints,
        parts,
        every: int = 1,
        agree: Callable[[bool], bool] = _own_decision,
    ):
        self.every = every
        self.seconds = minutes * 60.0
        self.checkpoints = checkpoints
        self.parts = parts
        self.agree = agree
        self.last = time.monotonic()

    def on_step_end(self, state: TrainState, out) -> None:
        now = time.monotonic()
        if self.agree(now - self.last >= self.seconds):
            self.checkpoints.save(state.step, self.parts(state))
            self.last = now
