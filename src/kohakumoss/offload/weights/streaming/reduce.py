"""The cross-rank sum of a host-stepped weight stream's block gradients on the last
micro-batch. See docs/internals/offload.md#weight-streaming."""

import contextlib
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor

from kohakumoss.offload.weights.streaming.device import DeviceStreams
from kohakumoss.offload.weights.streaming.grads import GradSink
from kohakumoss_common.calls import nothing


class BlockReducer:
    """Once all of block `i`'s last-micro-batch gradients have arrived (`collect`),
    its host sum plus them fill one fp32 flat buffer on a side stream, `reduce(flat,
    i)` sums it across ranks there (to the block's owner), and the result replaces the
    owner's host sum in `grads`. `reduce` is None until `bind`."""

    def __init__(
        self,
        grads: GradSink,
        dev: DeviceStreams,
        trainable: bool,
        mine: Callable[[int], bool],
        stats: dict,
    ):
        device, dm, side = dev.device, dev.dm, dev.streamed
        self.grads = grads
        self.params = grads.params
        self.device = device
        self.dm = dm
        self.streamed = side
        self.mine = mine
        self.stats = stats
        self.reduce: Callable[[Tensor, int], None] | None = None
        self.flat: Tensor | None = None
        self.pending: dict[int, dict[tuple[int, str], Tensor]] = {}
        self.red = dm.Stream(device) if side and trainable else None
        if side:
            self._order, self._side, self._keep = (
                BlockReducer._order_device,
                partial(dm.stream, self.red),
                BlockReducer._keep_device,
            )
        else:
            self._order, self._side, self._keep = (
                nothing,
                contextlib.nullcontext,
                nothing,
            )

    def _order_device(self) -> None:
        self.red.wait_stream(self.dm.current_stream(self.device))

    def _keep_device(self, used: list[Tensor]) -> None:
        for t in used:
            t.record_stream(self.red)

    def bind(self, reduce: Callable[[Tensor, int], None]):
        """Use `reduce`; the flat buffer (the largest block's fp32 gradients) is
        allocated here, once, on the current stream. Returns the side stream it runs on
        (None on the host)."""
        self.reduce = reduce
        if self.flat is None:
            largest = max(
                (
                    sum(self.params[k].numel() for k in keys)
                    for keys in self.grads.block_keys
                ),
                default=0,
            )
            self.flat = torch.empty(largest, dtype=torch.float32, device=self.device)
        return self.red

    def collect(self, key: tuple[int, str], grad: Tensor) -> None:
        pending = self.pending.setdefault(key[0], {})
        pending[key] = grad
        if len(pending) == len(self.grads.block_keys[key[0]]):
            self.reduce_block(key[0], self.pending.pop(key[0]))

    @torch.no_grad()
    def reduce_block(self, index: int, grads: dict) -> None:
        """Earlier micro-batches' host sum + this one's gradients -> the fp32 flat
        buffer on the side stream -> `reduce` -> the owner's `host_grad` (replacing the
        sum)."""
        sink = self.grads
        keys = sink.block_keys[index]
        sink.land_block(index, keys)
        self._order(self)
        used = list(grads.values())
        with self._side():
            total = sum(self.params[key].numel() for key in keys)
            flat = self.flat[:total]
            views, offset = [], 0
            for key in keys:
                shape = self.params[key].shape
                view = flat[offset : offset + shape.numel()].view(shape)
                offset += shape.numel()
                if sink.filled.get(key, False):
                    view.copy_(sink.host_grad[key], non_blocking=self.streamed)
                    view.add_(grads[key])
                else:
                    view.copy_(grads[key])
                views.append((key, view))
            if self.reduce is not None:
                self.reduce(flat, index)
            if self.mine(index):
                for key, view in views:
                    sink.buffer(key).copy_(view, non_blocking=self.streamed)
                    sink.filled[key] = True
                self.stats["bytes_d2h"] += total * 4
        self._keep(self, used)

    def synchronize(self) -> None:
        """Wait for every reduction queued on the side stream."""
        if self.red is not None:
            self.red.synchronize()
