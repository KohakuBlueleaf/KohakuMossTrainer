"""Checkpoint parts under FSDP2: full trainable / EMA tensors gathered on every rank
(rank 0 writes), the optimiser state as one full state dict."""

from kohakumoss.parallel.fsdp import (
    full_tensors,
    load_optimizer_state,
    optimizer_state,
    shard_like,
    unsharded,
)
from kohakumoss.training.drivers.common import Parts
from kohakumoss.training.state import TrainState


class FsdpParts(Parts):
    """Parts of an FSDP2-sharded `model`; every rank calls it (collectives)."""

    def __init__(self, *args, model, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def full_trainable(self) -> dict:
        """The trainable state as full host tensors (every rank)."""
        with unsharded(self.model):
            return {
                k: v.detach().to("cpu", copy=True)
                for k, v in self.trainable.state_dict().items()
            }

    def __call__(self, state: TrainState) -> dict:
        parts = super().__call__(state)
        parts["trainable"] = self.full_trainable()
        parts["ema"] = {
            k: v.cpu() for k, v in full_tensors(self.ema.state_dict()).items()
        }
        parts["optim"]["optimizer"] = optimizer_state(self.model, self.bundle.optimizer)
        return parts

    def restore(self, ckpt: dict, state: TrainState) -> None:
        ckpt = dict(ckpt)
        ckpt["trainable"] = shard_like(ckpt["trainable"], self.trainable.state_dict())
        ckpt["ema"] = shard_like(ckpt["ema"], self.ema.state_dict())
        full_optim = ckpt["optim"]["optimizer"]
        ckpt["optim"] = {
            **ckpt["optim"],
            "optimizer": self.bundle.optimizer.state_dict(),
        }
        super().restore(ckpt, state)
        load_optimizer_state(self.model, self.bundle.optimizer, full_optim)
