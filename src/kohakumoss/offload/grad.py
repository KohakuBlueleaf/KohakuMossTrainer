"""Gradient offload: per-parameter gradients summed in fp32 host memory across the
GRAD_ACC micro-batches, finished at `flush()` or inside the last backward
(`arm(last, final)`). See docs/internals/offload.md."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import module, streamed
from kohakumoss.offload.host_acc import HostAccumulator


@dataclass
class _Lane:
    host: Tensor
    filled: bool = False
    consumed: object = None


class _Done:
    def synchronize(self) -> None:
        return None


def _event_after(device: torch.device):
    """An event on `device`'s current stream (host: already complete)."""
    if not streamed(device):
        return _Done()
    dm = module(device)
    ev = dm.Event()
    ev.record(dm.current_stream(device))
    return ev


class _Ring:
    """Device staging for host sums coming back (H2D stream) and the update stream."""

    def __init__(self, device: torch.device, largest: int, slots: int = 3):
        self.dm = module(device)
        self.h2d = self.dm.Stream(device)
        self.opt = self.dm.Stream(device)
        self.buffers = [
            torch.empty(largest, dtype=torch.float32, device=device)
            for _ in range(slots)
        ]
        cur = self.dm.current_stream(device)
        self.free = [self.dm.Event() for _ in range(slots)]
        for ev in self.free:
            ev.record(cur)
        self.next = 0

    def take(self, numel: int) -> tuple[int, Tensor]:
        slot = self.next
        self.next = (slot + 1) % len(self.buffers)
        return slot, self.buffers[slot][:numel]


class GradOffload:
    """Post-accumulate-grad hooks moving gradients to host fp32 accumulators."""

    def __init__(self, params: Iterable[nn.Parameter], pinned: bool = True):
        self.params = [p for p in params if p.requires_grad]
        self.host: dict[nn.Parameter, Tensor] = {}
        self.lanes: dict[int, _Lane] = {}
        self.stats = {"bytes_d2h": 0, "bytes_h2d": 0}
        self._streams: dict[torch.device, object] = {}
        self.final: Callable | None = None
        self.handles = []
        self._mode = "offload"
        self.acc = HostAccumulator(pinned)
        self.rings: dict[torch.device, _Ring] = {}
        largest: dict[torch.device, int] = {}
        for p in self.params:
            largest[p.device] = max(largest.get(p.device, 0), p.numel())
        for p in self.params:
            cuda = streamed(p.device)
            host = torch.zeros(p.shape, dtype=torch.float32, pin_memory=pinned and cuda)
            lane = _Lane(host)
            if cuda:
                stream = self._stream(p.device)
                offload = partial(self._offload_device, lane, stream, id(p))
                ring = self.rings.get(p.device)
                if ring is None:
                    ring = _Ring(p.device, largest[p.device])
                    self.rings[p.device] = ring
                finish = partial(self._finish_device, lane, ring)
            else:
                offload = partial(self._offload_host, lane)
                finish = partial(self._finish, lane)
            self.host[p] = host
            self.lanes[id(p)] = lane
            hook = partial(self._hook, {"offload": offload, "finish": finish})
            self.handles.append(p.register_post_accumulate_grad_hook(hook))

    def _stream(self, device: torch.device):
        stream = self._streams.get(device)
        if stream is None:
            stream = module(device).Stream(device)
            self._streams[device] = stream
        return stream

    def arm(self, last: bool, final: Callable | None = None) -> None:
        """Offload this micro-batch's grads, or (last with `final`) finish them in place."""
        self.final = final
        self._mode = "finish" if last and final is not None else "offload"

    def _hook(self, bodies: dict, p: Tensor) -> None:
        bodies[self._mode](p)

    @torch.no_grad()
    def _offload_host(self, lane: _Lane, p: Tensor) -> None:
        lane.host.add_(p.grad)
        lane.filled = True
        p.grad = None

    @torch.no_grad()
    def _offload_device(self, lane: _Lane, stream, key: int, p: Tensor) -> None:
        g = p.grad
        self.acc.add(key, g, lane.host, stream)
        lane.filled = True
        self.stats["bytes_d2h"] += g.numel() * g.element_size()
        p.grad = None

    @torch.no_grad()
    def _finish(self, lane: _Lane, p: Tensor) -> None:
        if lane.filled:
            self.acc.wait(id(p))
            p.grad.add_(lane.host.to(p.device, non_blocking=True))
            self.stats["bytes_h2d"] += lane.host.numel() * 4
            lane.consumed = _event_after(p.device)
            lane.filled = False
        self.final(p)
        p.grad = None

    @torch.no_grad()
    def _finish_device(self, lane: _Lane, ring: "_Ring", p: Tensor) -> None:
        g = p.grad
        dm = ring.dm
        ring.opt.wait_stream(dm.current_stream(g.device))
        if lane.filled:
            self.acc.wait(id(p))
            slot, buf = ring.take(g.numel())
            ring.h2d.wait_event(ring.free[slot])
            with dm.stream(ring.h2d):
                buf.copy_(lane.host.view(-1), non_blocking=True)
            ev = dm.Event()
            ev.record(ring.h2d)
            ring.opt.wait_event(ev)
            with dm.stream(ring.opt):
                g.add_(buf.view(g.shape))
            ring.free[slot].record(ring.opt)
            lane.consumed = ev
            lane.filled = False
            self.stats["bytes_h2d"] += lane.host.numel() * 4
        with dm.stream(ring.opt):
            self.final(p)
        g.record_stream(ring.opt)
        p.grad = None

    def finish(self) -> None:
        """The compute stream waits for every in-backward update (before the next
        forward reads the parameters)."""
        for device, ring in self.rings.items():
            ring.dm.current_stream(device).wait_stream(ring.opt)

    def flush(self) -> None:
        """Land every in-flight copy (call after the last micro-batch's backward)."""
        self.acc.drain()

    def zero(self) -> None:
        """Clear the host accumulators (call after the optimiser step)."""
        self.acc.drain()
        for lane in self.lanes.values():
            if lane.consumed is not None:
                lane.consumed.synchronize()
                lane.consumed = None
            lane.host.zero_()
            lane.filled = False

    def restore(self, p: nn.Parameter) -> None:
        """Put the summed gradient of one parameter back on its device."""
        p.grad = self.host[p].to(p.device, p.dtype, non_blocking=True)
        self.lanes[id(p)].consumed = _event_after(p.device)
        self.stats["bytes_h2d"] += p.numel() * 4

    def to_device(self) -> None:
        """Put every summed gradient back on its device (for a device optimiser)."""
        for p in self.params:
            self.restore(p)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
