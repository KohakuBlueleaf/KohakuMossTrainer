"""BlockRunners: how the backbone loop executes block i (checkpoint, offload, stream).

A runner is composed once from the MemoryPlan; the loop calls `run` with no branching
on configuration. Sub-block checkpointing uses `block.sub_blocks()` when the block
provides it (callables `(x, ctx) -> x`), else the whole block.
"""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from kohakumoss.types.api import BlockCtx

SAVE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.linear.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}


def register_saved_op(op) -> None:
    """Make selective checkpointing keep the outputs of `op` (e.g. a custom attention)."""
    SAVE_OPS.add(op)


def _save_matmuls(ctx, op, *args, **kwargs) -> CheckpointPolicy:
    if op in SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_context():
    """Checkpoint contexts that keep matmul/attention outputs and recompute the rest."""
    return create_selective_checkpoint_contexts(_save_matmuls)


class PlainRunner:
    """`block(x, ctx)`."""

    def bind(self, layers) -> "PlainRunner":
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        return block(x, ctx)


class CheckpointRunner:
    """Non-reentrant activation checkpointing of whole blocks, sub-blocks or selectively.

    `blocks` restricts checkpointing to those indices (partial checkpointing).
    `bind(layers)` resolves each block's callable once; `run` only indexes it.
    """

    def __init__(self, mode: str = "block", blocks: frozenset[int] | None = None):
        if mode not in ("block", "sub-block", "selective"):
            raise ValueError(mode)
        self.mode = mode
        self.blocks = blocks
        self._ckpt = partial(checkpoint, use_reentrant=False)
        if mode == "selective":
            self._ckpt = partial(
                checkpoint, use_reentrant=False, context_fn=selective_context
            )
        self._fns: tuple[Callable, ...] = ()

    def _resolve(self, index: int, block: nn.Module) -> Callable:
        if self.blocks is not None and index not in self.blocks:
            return block
        if self.mode == "sub-block" and hasattr(block, "sub_blocks"):
            return partial(_run_parts, self._ckpt, tuple(block.sub_blocks()))
        return partial(self._ckpt, block)

    def bind(self, layers) -> "CheckpointRunner":
        self._fns = tuple(self._resolve(i, b) for i, b in enumerate(layers))
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        return self._fns[index](x, ctx)


def _run_parts(ckpt, parts, x, ctx):
    for part in parts:
        x = ckpt(part, x, ctx)
    return x


class OffloadRunner:
    """Wraps another runner; saved activations of each block go to pinned host memory."""

    def __init__(self, inner, offload):
        self.inner = inner
        self.offload = offload

    def bind(self, layers) -> "OffloadRunner":
        self.inner.bind(layers)
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        with self.offload.block(index):
            return self.inner.run(index, block, x, ctx)


class StreamRunner:
    """Wraps another (checkpointing) runner; block weights are streamed by `engine`."""

    def __init__(self, inner, engine):
        self.inner = inner
        self.engine = engine

    def bind(self, layers) -> "StreamRunner":
        self.inner.bind(layers)
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        x = self.engine.enter(index, x)
        y = self.inner.run(index, block, x, ctx)
        return self.engine.exit(index, y)


class StageRunner:
    """Runner of a pipeline stage holding layers [lo, hi); maps local -> global index."""

    def __init__(self, inner, offset: int):
        self.inner = inner
        self.offset = offset

    def bind(self, layers) -> "StageRunner":
        self.inner.bind(layers)
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        return self.inner.run(index + self.offset, block, x, ctx)


def checkpoint_blocks(depth: int, fraction: float) -> frozenset[int]:
    """The first round(depth * fraction) block indices (earliest blocks hold most)."""
    return frozenset(range(round(depth * fraction)))


def build_runner(
    checkpoint_value: str,
    layers,
    offload=None,
    engine=None,
) -> object:
    """Compose and bind the runner for a plan's checkpoint mode, offload and stream."""
    if checkpoint_value.startswith("partial:"):
        frac = float(checkpoint_value.split(":", 1)[1])
        runner = CheckpointRunner("block", checkpoint_blocks(len(layers), frac))
    elif checkpoint_value == "none":
        runner = PlainRunner()
    else:
        runner = CheckpointRunner(checkpoint_value)
    if offload is not None:
        runner = OffloadRunner(runner, offload)
    if engine is not None:
        runner = StreamRunner(runner, engine)
    return runner.bind(layers)
