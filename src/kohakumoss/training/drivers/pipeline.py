"""Pipeline drivers: the native loop with the step's micro-batches handed to a pipeline
engine (Linux only).

- `ThreadedDriver`: one process, stages on several devices (`ThreadedPipeline`).
- `PipelineDriver`: one process per stage (`PipelineEngine`, torch.distributed
  pipelining); every rank draws the same batches and noise, only activations cross
  stage boundaries, each rank steps the parameters of its own stages.
"""

import torch
import torch.distributed as dist

from kohakumoss.registry import import_object
from kohakumoss.training.drivers.common import Parts
from kohakumoss.training.drivers.native import NativeDriver
from kohakumoss.training.state import TrainState
from kohakumoss.types.step import StepOut

MICRO_BATCH = "kohakumoss.parallel.pipeline.MicroBatch"
BROADCAST_LOSS = "kohakumoss.parallel.pipeline.broadcast_loss"


class ThreadedDriver(NativeDriver):
    """GRAD_ACC micro-batches per step through a single-process multi-device pipeline;
    predictions come back to the first device for the loss."""

    def __init__(self, *args, pipe, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = pipe

    def run_group(self, step, group, count, bundle, state: TrainState):
        denom = self.normalizer.denominator(count, self.device)
        preps = [step.prepare(self.batcher(hb), state.step) for hb in group]
        scale = bundle.scaler.scale
        device = self.device

        def loss_fn(pred, p):
            return step.loss(pred.to(device), p, denom) * scale

        jobs = [((p.z, p.text, p.sigma, p.seq), p) for p in preps]
        with step.autocast():
            total = self.pipe.step(jobs, loss_fn).to(device) / scale
        bundle.observe(total)
        return StepOut(total, denom, {"sigma_mean": preps[-1].sigma.mean()})


class PipelineDriver(NativeDriver):
    """MICROBATCHES micro-batches per step through a process-group pipeline."""

    def __init__(self, *args, engine, world: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.world = world
        self.micro = import_object(MICRO_BATCH)
        self.broadcast = import_object(BROADCAST_LOSS)

    def run_group(self, step, group, count, bundle, state: TrainState):
        preps = [step.prepare(self.batcher(hb), state.step) for hb in group]
        mbs = [self._micro(hb, p) for hb, p in zip(group, preps)]
        with step.autocast():
            loss = self.engine.step(mbs, float(max(count, 1)))
        total = self.broadcast(loss, self.device, self.world) / bundle.scaler.scale
        bundle.observe(total)
        return StepOut(
            total, torch.tensor(count), {"sigma_mean": preps[-1].sigma.mean()}
        )

    def _micro(self, hb, p):
        """Engine micro-batch: device tensors plus the host layout (the grid and
        token-drop subset of the host batch, text lengths from the host text plan)."""
        text_len = torch.tensor(p.seq.attn_text.lengths, dtype=torch.int64)
        return self.micro(
            p.z,
            p.text,
            p.sigma,
            hb.hw,
            text_len,
            p.target,
            p.weight,
            img_keep=hb.img_keep,
        )


class PipelineObjective:
    """Engine loss of one micro-batch: token MSE sum on its image tokens (the kept ones
    under token drop), loss-scaled."""

    def __init__(self, token_loss, scaler):
        self.token_loss = token_loss
        self.scaler = scaler

    def __call__(self, pred, data):
        sizes = data.img_counts().to(pred.device)
        seg = torch.repeat_interleave(
            torch.arange(len(sizes), device=pred.device), sizes
        )
        total = self.token_loss(
            pred, data.target, seg.to(torch.int32), data.weight, None
        )
        return total * self.scaler.scale


def _gather_merge(full: dict, local: dict, world: int) -> dict:
    """Rank 0: `full` overwritten by every rank's `local` entries (others: `full`)."""
    out = [None] * world if dist.get_rank() == 0 else None
    dist.gather_object({k: v.detach().cpu() for k, v in local.items()}, out, dst=0)
    if out is None:
        return full
    merged = dict(full)
    for part in out:
        merged.update(part)
    return merged


class PipelineParts(Parts):
    """Checkpoint parts across stages: trainable / EMA entries of every rank merged on
    rank 0, optimiser state saved per rank."""

    def __init__(self, *args, local_names: set[str], rank: int, world: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_names = local_names
        self.rank = rank
        self.world = world

    def __call__(self, state: TrainState) -> dict:
        parts = super().__call__(state)
        full = parts["trainable"]
        local = {k: v for k, v in full.items() if k in self.local_names}
        parts["trainable"] = _gather_merge(full, local, self.world)
        parts["ema"] = _gather_merge(parts["ema"], parts["ema"], self.world)
        out = [None] * self.world if self.rank == 0 else None
        dist.gather_object(parts["optim"], out, dst=0)
        parts["optim"] = {"ranks": out}
        return parts

    def restore(self, ckpt: dict, state: TrainState) -> None:
        ckpt = dict(ckpt)
        ckpt["optim"] = ckpt["optim"]["ranks"][self.rank]
        super().restore(ckpt, state)
