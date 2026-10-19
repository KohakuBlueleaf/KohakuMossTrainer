"""Native driver: the plain training loop (the default driver)."""

from collections.abc import Callable

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers.common import (
    Parts,
    finish_step,
    grouped,
    last_flags,
    loss_count,
    micro_steps,
    no_accumulate,
)
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import image_token_count
from kohakumoss_common.calls import nothing


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
        end_step: Callable[[], None] = nothing,
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
            out = self.run_group(step, group, count, bundle, state, ema)
            self.end_step()
            finish_step(state, group, bundle, ema, self.runner, out)
        self.checkpoints.finish(state.step, self.parts(state))
        return state

    def run_group(
        self, step, group: list, count: int, bundle, state: TrainState, ema=None
    ):
        """Forward + backward of one optimiser step's micro-batches; the last StepOut."""
        denom = self.normalizer.denominator(count, self.device)
        return micro_steps(
            step,
            self.batcher,
            self.accumulate,
            self.lasts,
            bundle,
            group,
            denom,
            state.step,
            ema,
        )
