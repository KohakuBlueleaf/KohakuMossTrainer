"""BlockRunners: how the backbone loop executes block i (checkpoint, offload, stream).

A runner is composed once from the MemoryPlan; the loop calls `run` with no branching
on configuration. Sub-block checkpointing uses `block.sub_blocks()` when the block
provides it (callables `(x, ctx) -> x`), else the whole block.
"""

import contextlib
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from kohakumoss.logging.phases import RECOMPUTE, phase
from kohakumoss.model.runner import PlainRunner
from kohakumoss.types.context import BlockCtx
from kohakumoss_kernels.saved_ops import KEPT

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
    if op in SAVE_OPS or op in KEPT:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def selective_context():
    """Checkpoint contexts that keep matmul/attention outputs (aten, and the kernel
    attention / fused-linear forwards of `kohakumoss_kernels.saved_ops.KEPT`) and
    recompute the rest."""
    return create_selective_checkpoint_contexts(_save_matmuls)


class ForwardThreads:
    """Recompute context: the intra-op thread count of the forward (`threads`) for the
    recompute, the caller's restored after it, around an `inner` recompute context,
    inside a `recompute` profiler range. See
    docs/internals/adapter-binding.md#checkpoint-recompute."""

    def __init__(self, threads: int, inner=None):
        self.threads = threads
        self.inner = inner if inner is not None else contextlib.nullcontext()

    def __enter__(self):
        self.range = phase(RECOMPUTE)
        self.range.__enter__()
        self.prev = torch.get_num_threads()
        if self.prev != self.threads:
            torch.set_num_threads(self.threads)
        return self.inner.__enter__()

    def __exit__(self, *exc):
        try:
            return self.inner.__exit__(*exc)
        finally:
            if self.prev != self.threads:
                torch.set_num_threads(self.prev)
            self.range.__exit__(*exc)


def forward_state(inner: Callable | None = None) -> Callable:
    """A checkpoint `context_fn`: `inner`'s (forward, recompute) contexts (none by
    default), the recompute run under the forward's intra-op thread count."""

    def contexts():
        forward, recompute = inner() if inner is not None else (None, None)
        threads = torch.get_num_threads()
        forward = forward if forward is not None else contextlib.nullcontext()
        return forward, ForwardThreads(threads, recompute)

    return contexts


class CheckpointRunner:
    """Non-reentrant activation checkpointing of whole blocks, sub-blocks or selectively;
    each recompute runs under its forward's intra-op thread count (`forward_state`).

    `blocks` restricts checkpointing to those indices (partial checkpointing).
    `bind(layers)` resolves each block's callable once; `run` only indexes it.
    """

    def __init__(self, mode: str = "block", blocks: frozenset[int] | None = None):
        if mode not in ("block", "sub-block", "selective"):
            raise ValueError(mode)
        self.mode = mode
        self.blocks = blocks
        inner = selective_context if mode == "selective" else None
        self._ckpt = partial(
            checkpoint, use_reentrant=False, context_fn=forward_state(inner)
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
    """Wraps another (checkpointing) runner; block weights are streamed by
    `weight_stream`."""

    def __init__(self, inner, weight_stream):
        self.inner = inner
        self.weight_stream = weight_stream

    def bind(self, layers) -> "StreamRunner":
        self.inner.bind(layers)
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        x = self.weight_stream.enter(index, x)
        y = self.inner.run(index, block, x, ctx)
        return self.weight_stream.exit(index, y)


class StageRunner:
    """Runner of a pipeline stage holding layers [offset, offset + len(layers)); maps
    local -> global index. The inner runner binds at the global indices."""

    def __init__(self, inner, offset: int):
        self.inner = inner
        self.offset = offset

    def bind(self, layers) -> "StageRunner":
        self.inner.bind((None,) * self.offset + tuple(layers))
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
    weight_stream=None,
) -> object:
    """Compose and bind the runner for a plan's checkpoint mode, offload and weight
    stream."""
    if checkpoint_value.startswith("partial:"):
        frac = float(checkpoint_value.split(":", 1)[1])
        runner = CheckpointRunner("block", checkpoint_blocks(len(layers), frac))
    elif checkpoint_value == "none":
        runner = PlainRunner()
    else:
        runner = CheckpointRunner(checkpoint_value)
    if offload is not None:
        runner = OffloadRunner(runner, offload)
    if weight_stream is not None:
        runner = StreamRunner(runner, weight_stream)
    return runner.bind(layers)
