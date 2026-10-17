"""Lightning driver: the same TrainStep / OptimBundle under a LightningModule with
manual optimisation on this process's one device. Multi-process runs are launched and
gradient-synced by KohakuMoss (`accumulate` arms each micro-batch, as in the native
driver); Lightning only drives the loop."""

from collections.abc import Callable

import torch

from kohakumoss.training.callbacks.runner import CallbackRunner
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

try:
    import lightning as L
except ImportError:
    L = None

_Base = L.LightningModule if L is not None else torch.nn.Module


def _nothing() -> None:
    return None


class MossModule(_Base):
    """Wraps the backbone; `training_step` takes one group of GRAD_ACC host batches."""

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
    ):
        super().__init__()
        self.end_step = end_step
        self.automatic_optimization = False
        self.backbone = backbone
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
        denom = self.normalizer.denominator(count, self.device)
        out = None
        for hb, last in zip(group, self.lasts):
            batch = self.batcher(hb)
            with self.accumulate(last):
                out = self.step_fn(batch, denom, state.step)
                self.bundle.backward(out.loss)
        self.end_step()
        account(state, group)
        if self.bundle.step():
            self.ema.update(state.step)
        state.step += 1
        record(state, self.bundle)
        self.runner.run(state, out)
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
        end_step: Callable[[], None] = _nothing,
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
        if not self.checkpoints.saved(state.step):
            self.checkpoints.save(state.step, self.parts(state))
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
