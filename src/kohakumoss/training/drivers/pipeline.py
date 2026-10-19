"""Pipeline drivers: the native loop with the step's micro-batches handed to a pipeline
engine (Linux only).

- `ThreadedDriver`: one process, stages on several devices (`ThreadedPipeline`).
- `PipelineDriver`: one process per stage (`PipelineEngine`, torch.distributed
  pipelining); every rank draws the same batches and noise, only activations cross
  stage boundaries, each rank steps the parameters of its own stages.

Both draw each micro-batch's adapter step state (T-LoRA masks, DyLoRA ranks,
module-dropout draws) in micro-batch order, and the engine installs a micro-batch's
state again before its own forward and backward.
"""

from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor

from kohakumoss.parallel.pipeline.distributed import MicroBatch
from kohakumoss.training.drivers.common import Parts
from kohakumoss.training.drivers.native import NativeDriver
from kohakumoss.training.state import TrainState
from kohakumoss.types.step import StepOut
from kohakumoss_common.calls import nothing


def begin_and_capture(train_step, trainable, step: int, sigma: Tensor, seq):
    """Install the adapter step state of one micro-batch (`train_step.pre_forward`)
    and return it (`trainable.step_state()`) for its later forward / backward."""
    train_step.pre_forward(sigma, seq, step)
    return trainable.step_state()


class ThreadedDriver(NativeDriver):
    """MICROBATCHES micro-batches per step through a single-process multi-device pipeline;
    predictions come back to the first device for the loss."""

    def __init__(self, *args, pipe, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = pipe
        pipe.install = self.parts.trainable.set_step_state

    def run_group(self, step, group, count, bundle, state: TrainState, ema=None):
        """The step's micro-batches through the pipe; a background EMA update is
        joined before the pipe's backward writes the parameters."""
        denom = self.normalizer.denominator(count, self.device)
        preps = [step.noise(self.batcher(hb), state.step) for hb in group]
        trainable = self.parts.trainable
        states = [
            begin_and_capture(step, trainable, state.step, p.sigma, p.seq)
            for p in preps
        ]
        scale = bundle.scaler.scale
        device = self.device

        def loss_fn(pred, p):
            return step.loss(pred.to(device), p, denom) * scale

        jobs = [((p.z, p.text, p.sigma, p.seq), p) for p in preps]
        getattr(ema, "join", nothing)()
        with step.autocast():
            total = self.pipe.step(jobs, loss_fn, states).to(device) / scale
        bundle.observe(total)
        out = StepOut(total, denom, {"sigma_mean": preps[-1].sigma.mean()})
        return out.detached()


class PipelineDriver(NativeDriver):
    """MICROBATCHES micro-batches per step through a process-group pipeline."""

    def __init__(self, *args, engine, world: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.world = world
        self.micro = MicroBatch
        engine.install = self.parts.trainable.set_step_state

    def run_group(self, step, group, count, bundle, state: TrainState, ema=None):
        """The step's micro-batches through the engine; a background EMA update is
        joined before the engine's backward writes the parameters."""
        preps = [step.noise(self.batcher(hb), state.step) for hb in group]
        mbs = [self._micro(hb, p) for hb, p in zip(group, preps)]
        self.engine.capture = partial(
            begin_and_capture, step, self.parts.trainable, state.step
        )
        getattr(ema, "join", nothing)()
        with step.autocast():
            loss = self.engine.step(mbs, float(max(count, 1)))
        total = loss / bundle.scaler.scale
        bundle.observe(total)
        out = StepOut(
            total, torch.tensor(count), {"sigma_mean": preps[-1].sigma.mean()}
        )
        return out.detached()

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


def to_host(obj):
    """`obj` with every tensor in it (nested dicts / lists / tuples) detached on the
    host (what a rank hands to `gather_object`; see docs/guides/multi-gpu.md)."""
    match obj:
        case Tensor():
            return obj.detach().cpu()
        case dict():
            return {k: to_host(v) for k, v in obj.items()}
        case list():
            return [to_host(v) for v in obj]
        case tuple():
            return tuple(to_host(v) for v in obj)
    return obj


HOST_GROUP: list = []


def host_group():
    """A gloo group over every rank for object collectives, made once (collective on
    first use): gather_object on it moves the pickled bytes through host memory, not
    the device."""
    if not HOST_GROUP:
        HOST_GROUP.append(dist.new_group(backend="gloo"))
    return HOST_GROUP[0]


def _gather_merge(full: dict, local: dict, world: int) -> dict:
    """Rank 0: `full` overwritten by every rank's `local` entries (others: `full`)."""
    out = [None] * world if dist.get_rank() == 0 else None
    dist.gather_object(to_host(local), out, dst=0, group=host_group())
    if out is None:
        return full
    merged = dict(full)
    for part in out:
        merged.update(part)
    return merged


class PipelineParts(Parts):
    """Checkpoint parts across stages: trainable / EMA entries of every rank merged on
    rank 0, optimiser state (on the host) saved per rank; `before_export` gives rank
    0's trainable set every stage's trained values."""

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
        ema = parts["ema"]
        local_ema = {
            k: v for k, v in ema.items() if k.split("/", 1)[-1] in self.local_names
        }
        parts["ema"] = _gather_merge(ema, local_ema, self.world)
        out = [None] * self.world if self.rank == 0 else None
        dist.gather_object(to_host(parts["optim"]), out, dst=0, group=host_group())
        parts["optim"] = {"ranks": out}
        return parts

    def before_export(self) -> None:
        """Every rank's stage entries merged into rank 0's live trainable set
        (collective: every rank calls it)."""
        full = self.trainable.state_dict()
        local = {k: v for k, v in full.items() if k in self.local_names}
        merged = _gather_merge(full, local, self.world)
        if self.rank == 0:
            self.trainable.load_state_dict(merged)

    def restore(self, ckpt: dict, state: TrainState) -> None:
        ckpt = dict(ckpt)
        ckpt["optim"] = ckpt["optim"]["ranks"][self.rank]
        super().restore(ckpt, state)
