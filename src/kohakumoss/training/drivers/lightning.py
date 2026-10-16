"""Lightning driver: the same TrainStep / OptimBundle under a LightningModule with
manual optimisation (single device or Lightning DDP)."""

from collections.abc import Callable

import torch

from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.drivers.common import Parts, account, grouped, record
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import image_token_count

try:
    import lightning as L
except ImportError:
    L = None

_Base = L.LightningModule if L is not None else torch.nn.Module


class MossModule(_Base):
    """Wraps the backbone; `training_step` takes one group of GRAD_ACC host batches."""

    def __init__(
        self, backbone, step, bundle, ema, batcher, normalizer, runner, state, count_fn
    ):
        super().__init__()
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

    def configure_optimizers(self):
        return None

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def training_step(self, group, batch_idx):
        state = self.state
        count = account(state, group, self.count_fn)
        denom = self.normalizer.denominator(count, self.device)
        out = None
        for hb in group:
            out = self.step_fn(self.batcher(hb), denom, state.step)
            self.bundle.backward(out.loss)
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
        devices: int = 1,
        strategy: str = "auto",
        count_fn: Callable = image_token_count,
    ):
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
        self.devices = devices
        self.strategy = strategy
        self.count_fn = count_fn

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
        )
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=self.steps - state.step,
            num_sanity_val_steps=0,
            accelerator=self.device.type,
            devices=self.devices,
            strategy=self.strategy,
            precision="32-true",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            use_distributed_sampler=False,
        )
        trainer.fit(module, train_dataloaders=_Groups(loader, self.grad_acc))
        self.checkpoints.save(state.step, self.parts(state))
        return state


class _Groups:
    """Iterable of GRAD_ACC-sized host-batch groups for Lightning."""

    def __init__(self, loader, n: int):
        self.loader = loader
        self.n = n

    def __iter__(self):
        return grouped(self.loader, self.n)
