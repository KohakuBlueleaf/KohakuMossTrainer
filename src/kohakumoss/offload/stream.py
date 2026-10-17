"""Block weight streaming: MossLinear weights in host memory, a device ring of a few
blocks with prefetch on a side stream; trainable blocks keep fp32 host masters. See
docs/internals/offload.md (weight streaming)."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

from kohakumoss.model.linear import MossLinear
from kohakumoss.offload.accel import module, streamed, with_index
from kohakumoss.offload.host_acc import HostAccumulator
from kohakumoss.quant.weights import QuantWeight, dequantize


class StreamedWeight:
    """Frozen BaseWeight backed by the engine's ring slot for (block, key)."""

    def __init__(self, engine: "WeightStream", block: int, key: str, shape):
        self.engine = engine
        self.block = block
        self.key = key
        self.shape = torch.Size(shape)

    def weight(self, dtype: torch.dtype) -> Tensor:
        return self.engine.slot(self.block, self.key).to(dtype)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        return F.linear(x, self.weight(x.dtype), bias)

    def materialize(self) -> Tensor:
        return self.engine.host[self.block][self.key]


class StreamedParamWeight(StreamedWeight):
    """Trainable streamed weight: gradients flow to the host fp32 master."""

    def __init__(self, engine, block, key, shape, master: nn.Parameter):
        super().__init__(engine, block, key, shape)
        self.master = master

    def weight(self, dtype: torch.dtype) -> Tensor:
        w = self.engine.slot(self.block, self.key)
        return _SlotParam.apply(self.master, w, self.engine, (self.block, self.key)).to(
            dtype
        )

    def materialize(self) -> Tensor:
        return self.master.detach()


class StreamedQuantWeight(StreamedWeight):
    """Streamed int8/fp8 base: the ring carries `q` and `scale`, dequantized on use."""

    def weight(self, dtype: torch.dtype) -> Tensor:
        q = self.engine.slot(self.block, self.key + ":q")
        scale = self.engine.slot(self.block, self.key + ":scale")
        return dequantize(q, scale, dtype)

    def materialize(self) -> Tensor:
        host = self.engine.host[self.block]
        return dequantize(
            host[self.key + ":q"], host[self.key + ":scale"], torch.float32
        )


class _Enter(Function):
    @staticmethod
    def forward(ctx, x, engine, index):
        ctx.engine, ctx.index = engine, index
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.engine.release(ctx.index)
        return grad, None, None


class _Exit(Function):
    @staticmethod
    def forward(ctx, y, engine, index):
        ctx.engine, ctx.index = engine, index
        return y.view_as(y)

    @staticmethod
    def backward(ctx, grad):
        engine = ctx.engine
        engine.ensure(ctx.index)
        engine.prefetch(ctx.index - 1)
        engine.on_block_backward(ctx.index)
        return grad, None, None


class _SlotParam(Function):
    @staticmethod
    def forward(ctx, param, slot, engine, key):
        ctx.engine, ctx.key = engine, key
        return slot.view_as(slot)

    @staticmethod
    def backward(ctx, grad):
        ctx.engine.on_grad(ctx.key, grad)
        return None, None, None, None


def _noop_block(index: int) -> None:
    return None


class WeightStream:
    """Ring-buffer weight streaming over a sequence of identically shaped blocks."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        device: torch.device | str,
        resident: int = 2,
        pinned: bool = True,
        trainable: bool = False,
        dtype: torch.dtype | None = None,
    ):
        if resident < 2:
            raise ValueError("resident must be >= 2")
        self.device = torch.device(device)
        self.device = with_index(self.device)
        self.cuda = streamed(self.device)
        self.dm = module(self.device)
        self.pinned = pinned and self.cuda
        self.trainable = trainable
        self.n = len(blocks)
        self.resident = min(resident, self.n)
        self.host: list[dict[str, Tensor]] = []
        self.params: dict[tuple[int, str], nn.Parameter] = {}
        self.layout: dict[str, tuple[torch.Size, torch.dtype]] = {}
        for i, block in enumerate(blocks):
            self.host.append(self._take(i, block, dtype))
        self.slots = [
            {
                k: torch.empty(s, dtype=d, device=self.device)
                for k, (s, d) in self.layout.items()
            }
            for _ in range(self.resident)
        ]
        self.slot_of: dict[int, int] = {}
        self.held: list[int | None] = [None] * self.resident
        self.age = [0] * self.resident
        self.clock = 0
        self.active: int | None = None
        self.staging: dict[str, Tensor] = {}
        self.host_grad: dict[tuple[int, str], Tensor] = {}
        self.grad_filled: dict[tuple[int, str], bool] = {}
        self.grad_ids = {key: n for n, key in enumerate(self.params)}
        self.stats = {"loads": 0, "bytes_h2d": 0, "bytes_d2h": 0}
        self.on_block_backward = _noop_block
        self.optimizer = None
        self._bind_device()

    def _bind_device(self) -> None:
        if not self.cuda:
            self._copy, self._wait, self._mark_free = (
                self._copy_host,
                self._nowait,
                self._nowait,
            )
            self.accumulate = self._accumulate_host
            self.on_grad = self.accumulate
            return
        self.acc = HostAccumulator(self.pinned)
        self.h2d = self.dm.Stream(self.device)
        self.d2h = self.dm.Stream(self.device)
        cur = self.dm.current_stream(self.device)
        self.ready = [self.dm.Event() for _ in range(self.resident)]
        self.free = [self.dm.Event() for _ in range(self.resident)]
        self.host_ready = [self.dm.Event() for _ in range(self.n)]
        for ev in self.free + self.host_ready:
            ev.record(cur)
        self._copy = self._copy_pinned if self.pinned else self._copy_staged
        self._wait, self._mark_free = self._wait_ready, self._record_free
        self.accumulate = self._accumulate_device
        self.on_grad = self.accumulate

    def _host(self, t: Tensor, dtype=None) -> Tensor:
        host = t.detach().to("cpu", dtype or t.dtype).contiguous()
        return host.pin_memory() if self.pinned else host

    def _take(self, index: int, block: nn.Module, dtype) -> dict[str, Tensor]:
        out = {}
        for name, lin in block.named_modules():
            if not isinstance(lin, MossLinear):
                continue
            if isinstance(lin.base, QuantWeight):
                if self.trainable:
                    raise ValueError("a quantized base cannot be streamed trainably")
                for part, t in (("q", lin.base.q), ("scale", lin.base.scale)):
                    out[f"{name}:{part}"] = self._host(t)
                    self.layout.setdefault(f"{name}:{part}", (t.shape, t.dtype))
                lin.set_base(StreamedQuantWeight(self, index, name, lin.base.shape))
                continue
            w = lin.weight.detach()
            work = dtype or w.dtype
            out[name] = self._host(w, work)
            self.layout.setdefault(name, (w.shape, work))
            if self.trainable:
                master = nn.Parameter(self._host(w, torch.float32))
                lin.weight = master
                self.params[(index, name)] = master
                lin.set_base(StreamedParamWeight(self, index, name, w.shape, master))
                continue
            lin.weight = nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=self.device), requires_grad=False
            )
            lin.set_base(StreamedWeight(self, index, name, w.shape))
        return out

    def slot(self, block: int, key: str) -> Tensor:
        """The device copy of (block, key), loading it when it is not resident."""
        slot = self.slot_of.get(block)
        if slot is None:
            self.ensure(block)
            slot = self.slot_of[block]
        return self.slots[slot][key]

    def _victim(self) -> int:
        candidates = [
            s
            for s in range(self.resident)
            if self.held[s] is None or self.held[s] != self.active
        ]
        return min(candidates, key=lambda s: self.age[s])

    def _load(self, index: int, slot: int) -> None:
        old = self.held[slot]
        if old is not None:
            self.slot_of.pop(old, None)
        self.held[slot] = index
        self.slot_of[index] = slot
        self.clock += 1
        self.age[slot] = self.clock
        self.stats["loads"] += 1
        self._copy(self.host[index], self.slots[slot], slot, index)

    def _copy_host(self, src: dict, dst: dict, slot: int, index: int) -> None:
        for k, t in src.items():
            dst[k].copy_(t)

    def _copy_pinned(self, src: dict, dst: dict, slot: int, index: int) -> None:
        self.h2d.wait_event(self.free[slot])
        self.h2d.wait_event(self.host_ready[index])
        with self.dm.stream(self.h2d):
            for k, t in src.items():
                dst[k].copy_(t, non_blocking=True)
                self.stats["bytes_h2d"] += t.numel() * t.element_size()
        self.ready[slot].record(self.h2d)

    def _copy_staged(self, src: dict, dst: dict, slot: int, index: int) -> None:
        self.h2d.wait_event(self.free[slot])
        self.host_ready[index].synchronize()
        with self.dm.stream(self.h2d):
            for k, t in src.items():
                dst[k].copy_(self._stage(k, t), non_blocking=True)
                self.stats["bytes_h2d"] += t.numel() * t.element_size()
        self.ready[slot].record(self.h2d)

    def _stage(self, key: str, t: Tensor) -> Tensor:
        buf = self.staging.get(key)
        if buf is None:
            buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            self.staging[key] = buf
        self.h2d.synchronize()
        buf.copy_(t)
        return buf

    def _nowait(self, slot: int) -> None:
        return None

    def _wait_ready(self, slot: int) -> None:
        self.dm.current_stream(self.device).wait_event(self.ready[slot])

    def _record_free(self, slot: int) -> None:
        self.free[slot].record(self.dm.current_stream(self.device))

    def ensure(self, index: int) -> None:
        """Block `index` is resident and the compute stream waits for its copy."""
        slot = self.slot_of.get(index)
        if slot is None:
            slot = self._victim()
            self._load(index, slot)
        self.active = index
        self.clock += 1
        self.age[slot] = self.clock
        self._wait(slot)

    def prefetch(self, index: int) -> None:
        if 0 <= index < self.n and index not in self.slot_of:
            self._load(index, self._victim())

    def release(self, index: int) -> None:
        """The compute stream is done with block `index`; its slot may be overwritten."""
        slot = self.slot_of.get(index)
        if slot is not None:
            self._mark_free(slot)

    def enter(self, index: int, x: Tensor) -> Tensor:
        self.ensure(index)
        self.prefetch(index + 1)
        return _Enter.apply(x, self, index)

    def exit(self, index: int, y: Tensor) -> Tensor:
        self.release(index)
        return _Exit.apply(y, self, index)

    def _grad_buffer(self, key: tuple[int, str]) -> Tensor:
        buf = self.host_grad.get(key)
        if buf is None:
            master = self.params[key]
            buf = torch.zeros(master.shape, dtype=torch.float32, pin_memory=self.pinned)
            self.host_grad[key] = buf
        return buf

    def _accumulate_host(self, key: tuple[int, str], grad: Tensor) -> None:
        self._grad_buffer(key).add_(grad.detach().float())
        self.grad_filled[key] = True

    def _accumulate_device(self, key: tuple[int, str], grad: Tensor) -> None:
        """Queue `host_grad[key] += grad` (worker thread; backward does not wait)."""
        self.acc.add(self.grad_ids[key], grad, self._grad_buffer(key), self.d2h)
        self.grad_filled[key] = True
        self.stats["bytes_d2h"] += grad.numel() * grad.element_size()

    def _drain(self) -> None:
        if self.cuda:
            self.acc.drain()

    def flush_grads(self) -> None:
        """Land every queued gradient and expose it as the host master's `.grad`."""
        self._drain()
        for key, filled in self.grad_filled.items():
            if filled:
                self.params[key].grad = self.host_grad[key]

    def land_block(self, index: int, keys) -> None:
        """Every queued gradient of one block's `keys` has landed in `host_grad`."""
        if self.cuda:
            for key in keys:
                self.acc.wait(self.grad_ids[key])

    def zero_grads(self) -> None:
        """Clear the host gradient sums (after the optimiser step)."""
        self._drain()
        for key, filled in self.grad_filled.items():
            if filled:
                self.host_grad[key].zero_()
                self.grad_filled[key] = False

    @torch.no_grad()
    def refresh(self) -> None:
        """Copy updated fp32 masters into the host streaming copies (after the step)."""
        for (index, key), param in self.params.items():
            self.host[index][key].copy_(param.detach())
        self.slot_of.clear()
        self.held = [None] * self.resident

    def storages(self) -> list[Tensor]:
        """Ring slot tensors (to exclude from activation offload)."""
        return [t for slot in self.slots for t in slot.values()]
