"""Native driver: the plain training loop (the default driver).

Everything fixed after build is prepared in `__init__`: callbacks and their cadence
(CallbackRunner, which also carries checkpoint saving), accumulation flags, the token
counter and the gradient-sync context. The loop only branches on the bundle's skip
result.
"""

from collections.abc import Callable

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers.common import (
    Parts,
    account,
    grouped,
    last_flags,
    loss_count,
    no_accumulate,
    record,
)
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import image_token_count


def _nothing() -> None:
    return None


class NativeDriver:
    """GRAD_ACC host batches per optimiser step, normalised by their global token count.
    `end_step` runs after the step's last backward (the adapter set's `end_step`)."""

    def __init__(
        self,
        steps: int,
        grad_acc: int,
        batcher: Callable,
        normalizer: TokenNormalizer,
        device,
        runner: CallbackRunner,
        checkpoints: CheckpointManager,
        parts: Parts,
        accumulate: Callable = no_accumulate,
        count_fn: Callable = image_token_count,
        end_step: Callable[[], None] = _nothing,
    ):
        self.end_step = end_step
        self.steps = steps
        self.grad_acc = grad_acc
        self.batcher = batcher
        self.normalizer = normalizer
        self.device = device
        self.runner = runner
        self.checkpoints = checkpoints
        self.parts = parts
        self.accumulate = accumulate
        self.count_fn = count_fn
        self.lasts = last_flags(grad_acc)

    def run(self, step, loader, bundle, ema, state: TrainState) -> TrainState:
        groups = grouped(loader, self.grad_acc)
        self.runner.reset(state.step)
        out = None
        while state.step < self.steps:
            group = next(groups)
            count = loss_count(group, self.count_fn)
            out = self.run_group(step, group, count, bundle, state)
            self.end_step()
            account(state, group)
            if bundle.step():
                ema.update(state.step)
            state.step += 1
            record(state, bundle)
            self.runner.run(state, out)
        if not self.checkpoints.saved(state.step):
            self.checkpoints.save(state.step, self.parts(state))
        return state

    def run_group(self, step, group: list, count: int, bundle, state: TrainState):
        """Forward + backward of one optimiser step's micro-batches; the last StepOut."""
        denom = self.normalizer.denominator(count, self.device)
        out = None
        for hb, last in zip(group, self.lasts):
            batch = self.batcher(hb)
            with self.accumulate(last):
                out = step(batch, denom, state.step)
                bundle.backward(out.loss)
        return out
