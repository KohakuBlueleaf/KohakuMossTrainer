"""Block weight streaming: MossLinear weights in host memory, a device ring of a few
blocks with prefetch on a side stream; trainable blocks keep fp32 host masters (shared
by the ranks of one node under DDP). See docs/internals/offload.md (weight streaming).
"""

from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.offload.accel import module, streamed, with_index
from kohakumoss.offload.host.arena import HostArena
from kohakumoss.offload.weights.layers import host_copy
from kohakumoss.offload.weights.shared import aligned, nbytes_of
from kohakumoss.offload.weights.streamed_weights import (
    StreamedComputeWeight,
    StreamedParamWeight,
    StreamedQuantWeight,
    StreamedWeight,
    _Enter,
    _Exit,
)
from kohakumoss.offload.weights.streaming import (
    BlockReducer,
    DeviceStreams,
    GradSink,
    H2DCopier,
    SlotRing,
    SpanTimer,
    read_timing,
)
from kohakumoss.quant.compute import ComputeWeight
from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss.quant.weights import QuantWeight
from kohakumoss_common.calls import nothing
from kohakumoss_common.hostmem import trim_heap

__all__ = [
    "StreamedComputeWeight",
    "StreamedParamWeight",
    "StreamedQuantWeight",
    "StreamedWeight",
    "WeightStream",
    "read_timing",
]


def steps_block(owners: list[int] | None, rank: int, index: int) -> bool:
    """Rank `rank` steps block `index` under `owners` (every block when None)."""
    return owners is None or owners[index] == rank


class WeightStream:
    """Ring-buffer weight streaming over a sequence of identically shaped blocks. A
    frozen ring keeps its first `slots - 2` blocks on the device once loaded and
    cycles the rest through two slots; a trainable (`swap`) ring cycles every block.

    Trainable rings: `share(nbytes)` (DDP on one node) returns the SharedHost region the
    fp32 masters and streaming copies live in; `owners[i]` is the rank that steps block
    i (its host gradient and optimiser state live there only).

    The stream owns the host copies (`host`, `params`, `layout`) and its parts: `ring`
    (SlotRing), `copier` (H2DCopier), `timer` (SpanTimer), `grads` (GradSink) and
    `reducer` (BlockReducer); `stats` counts loads, bytes and host syncs over all.
    `optimizer` is a weak reference to the in-backward block optimiser (StreamedAdamW),
    None without one."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        device: torch.device | str,
        slots: int = 2,
        pinned: bool = True,
        trainable: bool = False,
        dtype: torch.dtype | None = None,
        share: Callable | None = None,
        owners: list[int] | None = None,
        rank: int = 0,
    ):
        if slots < 2:
            raise ValueError("slots must be >= 2")
        self.device = with_index(torch.device(device))
        self.streamed = streamed(self.device)
        self.dm = module(self.device)
        self.pinned = pinned and self.streamed
        self.trainable = trainable
        self.owners = owners
        self.rank = rank
        self.n = len(blocks)
        slots = min(slots, self.n)
        kept = self.n if slots == self.n else slots - 2
        self.arena = None
        if trainable and share is not None:
            self.arena = share(self._shared_bytes(blocks, dtype))
        self.host: list[dict[str, Tensor]] = []
        self.params: dict[tuple[int, str], nn.Parameter] = {}
        self.layout: dict[str, tuple[torch.Size, torch.dtype]] = {}
        self.host_arena = None
        if trainable:
            for i, block in enumerate(blocks):
                self.host.append(self._take(i, block, dtype))
        else:
            self._take_frozen(blocks, dtype)
        if self.arena is not None:
            self.arena.ready()
        host_pinned = self.pinned
        if self.arena is not None:
            host_pinned = self.arena.pinned
        elif self.host_arena is not None:
            host_pinned = self.host_arena.pinned
        self.stats = {"loads": 0, "bytes_h2d": 0, "bytes_d2h": 0, "host_syncs": 0}
        dev = DeviceStreams(self.device, self.dm, self.streamed)
        self.timer = SpanTimer(dev, self.stats)
        self.copier = H2DCopier(
            dev, slots, self.n, host_pinned, trainable, self.stats, self.timer
        )
        keep = 0 if trainable else kept
        self.ring = SlotRing(
            self.host, self.layout, self.device, slots, keep, self.copier, self.stats
        )
        self.grads = GradSink(self.params, self.n, dev, self.pinned, self.stats)
        mine = partial(steps_block, owners, rank)
        self.reducer = BlockReducer(self.grads, dev, trainable, mine, self.stats)
        # the streamed weights' per-linear calls go straight to the parts
        self.slot, self.ensure = self.ring.slot, self.ring.ensure
        self.prefetch, self.release = self.ring.prefetch, self.ring.release
        self.land_block = self.grads.land_block
        self.on_grad = self.grads.accumulate
        self.on_block_backward = nothing
        self.optimizer = None
        self._sync = (
            partial(self.dm.synchronize, self.device) if self.streamed else nothing
        )

    @property
    def slots(self) -> int:
        """Device slots of the ring."""
        return self.ring.n_slots

    @property
    def accumulate(self) -> Callable[[tuple[int, str], Tensor], None]:
        """The host-sum gradient sink (`on_grad` outside a reduced last micro-batch)."""
        return self.grads.accumulate

    @property
    def host_grad(self) -> dict[tuple[int, str], Tensor]:
        return self.grads.host_grad

    @property
    def grad_filled(self) -> dict[tuple[int, str], bool]:
        return self.grads.filled

    @property
    def host_ready(self) -> list:
        """Per block, the event of its master write-back (device rings only)."""
        return self.copier.host_ready

    @property
    def flat(self) -> Tensor | None:
        """The cross-rank reduction buffer (`reduce_with`)."""
        return self.reducer.flat

    def _host(self, t: Tensor, dtype=None) -> Tensor:
        return host_copy(t, self.pinned, dtype)

    @staticmethod
    def _linears(block: nn.Module) -> list[tuple[str, MossLinear]]:
        return [(n, m) for n, m in block.named_modules() if isinstance(m, MossLinear)]

    def _shared_bytes(self, blocks, dtype) -> int:
        """Bytes of the shared region: every block's fp32 masters and streaming copies,
        in `_take` order."""
        total = 0
        for block in blocks:
            for _, lin in self._linears(block):
                w = lin.weight
                total += aligned(nbytes_of(w.shape, dtype or w.dtype))
                total += aligned(nbytes_of(w.shape, torch.float32))
        return total

    def mine(self, index: int) -> bool:
        """This rank steps block `index` (every block without owners)."""
        return steps_block(self.owners, self.rank, index)

    def _shared_pair(self, w: Tensor, work: torch.dtype) -> tuple[Tensor, Tensor]:
        """(streaming copy, fp32 master) views of the shared region, filled by rank 0."""
        copy = self.arena.take(w.shape, work)
        master = self.arena.take(w.shape, torch.float32)
        if self.arena.writer:
            src = w.detach().to("cpu")
            copy.copy_(src)
            master.copy_(src)
        return copy, master

    def _take(self, index: int, block: nn.Module, dtype) -> dict[str, Tensor]:
        out = {}
        for name, lin in self._linears(block):
            if isinstance(lin.base, W4A8Weight):
                lin.set_base(lin.base.to_int8())
            if isinstance(lin.base, QuantWeight):
                if self.trainable:
                    raise ValueError("a quantized base cannot be streamed trainably")
                for part, t in (("q", lin.base.q), ("scale", lin.base.scale)):
                    out[f"{name}:{part}"] = self._host(t)
                    self.layout.setdefault(f"{name}:{part}", (t.shape, t.dtype))
                base = StreamedQuantWeight(
                    self, index, name, lin.base.shape, lin.base.rotation
                )
                lin.set_base(base)
                continue
            if isinstance(lin.base, ComputeWeight):
                if self.trainable:
                    raise ValueError("a quantized base cannot be streamed trainably")
                for part, t in lin.base.tensors().items():
                    out[f"{name}:{part}"] = self._host(t)
                    self.layout.setdefault(f"{name}:{part}", (t.shape, t.dtype))
                base = StreamedComputeWeight(self, index, name, lin.base, dtype)
                lin.set_base(base)
                continue
            w = lin.weight.detach()
            work = dtype or w.dtype
            self.layout.setdefault(name, (w.shape, work))
            if self.trainable:
                if self.arena is not None:
                    out[name], data = self._shared_pair(w, work)
                else:
                    out[name], data = self._host(w, work), self._host(w, torch.float32)
                master = nn.Parameter(data)
                lin.weight = master
                self.params[(index, name)] = master
                lin.set_base(StreamedParamWeight(self, index, name, w.shape, master))
                continue
            out[name] = self._host(w, work)
            lin.weight = nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=self.device), requires_grad=False
            )
            lin.set_base(StreamedWeight(self, index, name, w.shape))
        return out

    @staticmethod
    def _frozen_parts(name: str, lin: MossLinear, dtype) -> list[tuple]:
        """(host key, tensor, host dtype) of each part a frozen linear streams: a
        quantized base's stored parts (a W4A8 base at its int8 decode), else the dense
        weight in `dtype`."""
        if isinstance(lin.base, W4A8Weight):
            lin.set_base(lin.base.to_int8())
        if isinstance(lin.base, QuantWeight):
            parts = (("q", lin.base.q), ("scale", lin.base.scale))
            return [(f"{name}:{p}", t, t.dtype) for p, t in parts]
        if isinstance(lin.base, ComputeWeight):
            parts = lin.base.tensors().items()
            return [(f"{name}:{p}", t, t.dtype) for p, t in parts]
        w = lin.weight.detach()
        return [(name, w, dtype or w.dtype)]

    def _take_frozen(self, blocks: Sequence[nn.Module], dtype) -> None:
        """Every frozen block's streamed parts copied into one HostArena (`host_arena`,
        exact size, pinned when the stream is), each linear's base re-pointed at its
        views as it goes, so the host sources are freed while the arena fills (the C
        heap's freed pages returned after each block)."""
        linears = [
            (i, name, lin, self._frozen_parts(name, lin, dtype))
            for i, block in enumerate(blocks)
            for name, lin in self._linears(block)
        ]
        specs = [(t.shape, d) for *_, parts in linears for _, t, d in parts]
        self.host_arena = arena = HostArena(specs, self.pinned, self.device)
        self.host = [{} for _ in blocks]
        k = 0
        for j in range(len(linears)):
            index, name, lin, parts = linears[j]
            linears[j] = None
            for key, t, d in parts:
                self.host[index][key] = arena.fill(k, t)
                self.layout.setdefault(key, (t.shape, d))
                k += 1
            parts = t = None
            self._rebase(index, name, lin, dtype)
            if j + 1 == len(linears) or linears[j + 1][0] != index:
                trim_heap()
        arena.finish()

    def _rebase(self, index: int, name: str, lin: MossLinear, dtype) -> None:
        """Point a frozen linear at its streamed form (its host parts are taken)."""
        if isinstance(lin.base, QuantWeight):
            shape, rotation = lin.base.shape, lin.base.rotation
            lin.set_base(StreamedQuantWeight(self, index, name, shape, rotation))
            return
        if isinstance(lin.base, ComputeWeight):
            lin.set_base(StreamedComputeWeight(self, index, name, lin.base, dtype))
            return
        shape = lin.weight.shape
        lin.weight = nn.Parameter(
            torch.empty(0, dtype=lin.weight.dtype, device=self.device),
            requires_grad=False,
        )
        lin.set_base(StreamedWeight(self, index, name, shape))

    def set_timing(self, on: bool) -> None:
        """Record copy and copy-wait spans on the device (read by `take_timing`)."""
        self.timer.set(on)

    def take_timing(self) -> dict[str, float]:
        """Seconds the H2D stream spent copying and the compute stream spent waiting
        for a copy since the last call, plus the bytes moved and host syncs taken;
        synchronizes the device (call once per sampler / train step)."""
        self._sync()
        return read_timing(self.cut_timing())

    def cut_timing(self) -> dict:
        """The copy / wait spans, bytes and host syncs since the last cut, unread (no
        device sync; `read_timing` once their events have completed)."""
        return self.timer.cut()

    def enter(self, index: int, x: Tensor) -> Tensor:
        self.ensure(index)
        self.prefetch(index + 1)
        return _Enter.apply(x, self, index)

    def exit(self, index: int, y: Tensor) -> Tensor:
        self.release(index)
        return _Exit.apply(y, self, index)

    def expose_grads(self) -> None:
        """Land every queued gradient and expose it as the host master's `.grad`."""
        self.grads.drain()
        self.reducer.synchronize()
        self.grads.expose(self.mine)

    def reduce_with(self, reduce: Callable[[Tensor, int], None]):
        """Sum block `i`'s gradient across ranks on the last micro-batch before it
        leaves for the host: `reduce(flat, i)` reduces in place on the current stream
        (to `owners[i]`). The flat buffer (one block's fp32 gradients) is allocated
        here, once, on the current stream. Returns the side stream it runs on (None on
        the host)."""
        return self.reducer.bind(reduce)

    def start(self, last: bool) -> None:
        """Before each micro-batch of a host-stepped ring: under a cross-rank reduction
        (`reduce_with`) the last one sends each block's step gradient through it;
        otherwise gradients accumulate on the host."""
        reducing = self.reducer.reduce is not None
        self.on_grad = self.reducer.collect if last and reducing else self.accumulate

    def zero_grads(self) -> None:
        """End the step's host gradient sums (after the optimiser step): the next
        gradient of each key replaces its sum, nothing is cleared."""
        self.grads.zero()

    @torch.no_grad()
    def refresh(self) -> None:
        """Copy updated fp32 masters of this rank's blocks into the host streaming
        copies (after the step); with a shared region every rank then sees them."""
        for (index, key), param in self.params.items():
            if self.mine(index):
                self.host[index][key].copy_(param.detach())
        self.ring.forget()
        if self.arena is not None:
            self.arena.ready()

    def storages(self) -> list[Tensor]:
        """Ring slot tensors (to exclude from activation offload)."""
        return self.ring.storages()

    def close(self) -> None:
        """Stop the gradient accumulator's worker thread (device streams only) and
        unpin the shared region."""
        self.grads.close()
        if self.arena is not None:
            self.arena.close()
