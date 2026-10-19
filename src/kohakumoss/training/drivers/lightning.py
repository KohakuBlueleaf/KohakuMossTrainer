"""Lightning driver: the same TrainStep / OptimBundle under a LightningModule with
manual optimisation on this process's one device. Multi-process runs are launched and
gradient-synced by KohakuMoss (`accumulate` starts each micro-batch, as in the native
driver); Lightning only drives the loop. The backbone is held outside the module's
tree, so Lightning's device moves (fit start, teardown) never touch the memory plan's
placement (host masters, streamed weights, offloaded state)."""

from collections.abc import Callable

import torch

from kohakumoss.training.callbacks.runner import CallbackRunner
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

try:
    import lightning as L
except ImportError:
    L = None

_Base = L.LightningModule if L is not None else torch.nn.Module


class MossModule(_Base):
    """Drives the step; `training_step` takes one group of GRAD_ACC host batches. The
    backbone (`self.held[0]`) and the step are plain attributes, not submodules: the
    module has no parameter or buffer of its own."""

    def __init__(
        self,
        backbone,
        step,
        bundle,
        ema,
        batcher,
        normalizer,
        runner,
        state,
        count_fn,
        accumulate,
        lasts,
        end_step,
        device=None,
    ):
        super().__init__()
        self.end_step = end_step
        self.automatic_optimization = False
        self.held = [backbone]
        self.run_device = device
        self.step_fn = step
        self.bundle = bundle
        self.ema = ema
        self.batcher = batcher
        self.normalizer = normalizer
        self.runner = runner
        self.state = state
        self.count_fn = count_fn
        self.accumulate = accumulate
        self.lasts = lasts

    def configure_optimizers(self):
        return None

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, group, batch_idx):
        state = self.state
        count = loss_count(group, self.count_fn)
        denom = self.normalizer.denominator(count, self.run_device)
        out = micro_steps(
            self.step_fn,
            self.batcher,
            self.accumulate,
            self.lasts,
            self.bundle,
            group,
            denom,
            state.step,
            self.ema,
        )
        self.end_step()
        finish_step(state, group, self.bundle, self.ema, self.runner, out)
        return None


class LightningDriver:
    """Builds a Trainer with Lightning precision off (KohakuMoss autocasts itself)."""

    def __init__(
        self,
        steps: int,
        grad_acc: int,
        batcher: Callable,
        normalizer,
        device,
        runner: CallbackRunner,
        checkpoints,
        parts: Parts,
        accumulate: Callable = no_accumulate,
        count_fn: Callable = image_token_count,
        end_step: Callable[[], None] = nothing,
    ):
        self.end_step = end_step
        if L is None:
            raise SystemExit(
                "DRIVER='lightning' needs the `lightning` package installed"
            )
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
        self.runner.reset(state.step)
        module = MossModule(
            step.backbone,
            step,
            bundle,
            ema,
            self.batcher,
            self.normalizer,
            self.runner,
            state,
            self.count_fn,
            self.accumulate,
            self.lasts,
            self.end_step,
            self.device,
        )
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=self.steps - state.step,
            num_sanity_val_steps=0,
            accelerator=self.device.type,
            devices=_one_device(self.device),
            strategy="auto",
            precision="32-true",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            use_distributed_sampler=False,
        )
        trainer.fit(module, train_dataloaders=_Groups(loader, self.grad_acc))
        self.checkpoints.finish(state.step, self.parts(state))
        return state


def _one_device(device: torch.device):
    """Lightning's `devices` for exactly this process's device."""
    return [device.index] if device.index is not None else 1


class _Groups:
    """Iterable of GRAD_ACC-sized host-batch groups for Lightning."""

    def __init__(self, loader, n: int):
        self.loader = loader
        self.n = n

    def __iter__(self):
        return grouped(self.loader, self.n)
