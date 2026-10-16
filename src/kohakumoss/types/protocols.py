"""Protocols the swappable components implement."""

from collections.abc import Iterator
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from kohakumoss.types.batch import DeviceBatch
from kohakumoss.types.seqinfo import SeqInfo
from kohakumoss.types.step import StepOut


class BaseWeight(Protocol):
    """Storage of a frozen or trainable base weight (dense, quantized or streamed)."""

    shape: torch.Size

    def weight(self, dtype: torch.dtype) -> Tensor: ...

    def materialize(self) -> Tensor: ...


class BlockRunner(Protocol):
    """Executes block i of the backbone loop (plain, checkpointed, streamed, staged)."""

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: Any) -> Tensor: ...


class TimestepSampler(Protocol):
    """Draws sigma in [0, 1] per sample."""

    def sample(self, batch: int, generator: torch.Generator, step: int) -> Tensor: ...


class Objective(Protocol):
    """Loss from prediction, target and layout; returns (fp32 sum, token count)."""

    def __call__(
        self, pred: Tensor, target: Tensor, seq: SeqInfo, weight: Tensor, mask
    ) -> tuple[Tensor, Tensor]: ...


class Step(Protocol):
    """Driver-independent training step."""

    def __call__(self, batch: DeviceBatch) -> StepOut: ...


class Callback(Protocol):
    """Hooks the drivers call; every hook is optional."""

    def on_step_end(self, state: Any, out: StepOut) -> None: ...


class Logger(Protocol):
    """Metric sink (rank 0)."""

    def log(self, metrics: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


class Driver(Protocol):
    """Owns the loop: optimiser, schedule, scaler, EMA, callbacks, checkpoints."""

    def run(self, step: Step, loader: Iterator, **parts) -> None: ...
