"""Gradient offload: per-parameter gradients summed in fp32 host memory across the
GRAD_ACC micro-batches, finished at `drain()` or inside the last backward
(`start(last, final)`). A step's first gradient of a parameter is copied into its sum
and later ones added, so a sum is never zeroed; `filled` marks the sums holding this
step's gradients. See docs/internals/offload.md."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import DONE, module, streamed
from kohakumoss.offload.optim.host_acc import HostAccumulator
from kohakumoss_common.weak import weak_call


@dataclass
class _Lane:
    host: Tensor
    filled: bool = False
    consumed: object = None


def _event_after(device: torch.device):
    """An event on `device`'s current stream (host: already complete)."""
    if not streamed(device):
        return DONE
    dm = module(device)
    ev = dm.Event()
    ev.record(dm.current_stream(device))
    return ev


RING_SLOTS = 3


def grad_ring_bytes(largest: int) -> int:
    """Device bytes of the staging ring of a device's largest offloaded parameter."""
    return RING_SLOTS * 4 * largest


class _Ring:
    """Device staging for host sums coming back (H2D stream) and the update stream."""

    def __init__(self, device: torch.device, largest: int, slots: int = RING_SLOTS):
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
        self.finishes: dict[int, Callable] = {}
        self.stats = {"bytes_d2h": 0, "bytes_h2d": 0}
        self._streams: dict[torch.device, object] = {}
        self.final: Callable | None = None
        self.handles = []
        self._mode = "offload"
        self.acc = HostAccumulator(pinned, stats=self.stats)
        self.rings: dict[torch.device, _Ring] = {}
        largest: dict[torch.device, int] = {}
        for p in self.params:
            largest[p.device] = max(largest.get(p.device, 0), p.numel())
        for p in self.params:
            side = streamed(p.device)
            host = torch.zeros(p.shape, dtype=torch.float32, pin_memory=pinned and side)
            lane = _Lane(host)
            if side:
                stream = self._stream(p.device)
                offload = partial(
                    GradOffload._offload_device, lane=lane, stream=stream, key=id(p)
                )
                ring = self.rings.get(p.device)
                if ring is None:
                    ring = _Ring(p.device, largest[p.device])
                    self.rings[p.device] = ring
                finish = partial(GradOffload._finish_device, lane=lane, ring=ring)
            else:
                offload = partial(GradOffload._offload_host, lane=lane)
                finish = partial(GradOffload._finish, lane=lane)
            self.host[p] = host
            self.lanes[id(p)] = lane
            self.finishes[id(p)] = finish
            bodies = {"offload": offload, "finish": finish}
            hook = weak_call(GradOffload._hook, self, bodies)
            self.handles.append(p.register_post_accumulate_grad_hook(hook))

    def _stream(self, device: torch.device):
        stream = self._streams.get(device)
        if stream is None:
            stream = module(device).Stream(device)
            self._streams[device] = stream
        return stream

    def start(self, last: bool, final: Callable | None = None) -> None:
        """Offload this micro-batch's grads, or (last with `final`) finish them in place."""
        self.final = final
        self._mode = "finish" if last and final is not None else "offload"

    def _hook(self, bodies: dict, p: Tensor) -> None:
        bodies[self._mode](self, p=p)

    @torch.no_grad()
    def _offload_host(self, lane: _Lane, p: Tensor) -> None:
        if lane.filled:
            lane.host.add_(p.grad)
        else:
            lane.host.copy_(p.grad)
        lane.filled = True
        p.grad = None

    @torch.no_grad()
    def _offload_device(self, lane: _Lane, stream, key: int, p: Tensor) -> None:
        g = p.grad
        self.acc.add(key, g, lane.host, stream, first=not lane.filled)
        lane.filled = True
        self.stats["bytes_d2h"] += g.nbytes
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

    @torch.no_grad()
    def finish_unhooked(self) -> None:
        """After the last backward started with a `final`: finish every parameter whose
        hook did not fire in it but whose sum holds earlier micro-batches' gradients
        (their sum becomes its gradient)."""
        if self._mode != "finish":
            return
        for p in self.params:
            if self.lanes[id(p)].filled:
                p.grad = torch.zeros_like(p)
                self.finishes[id(p)](self, p=p)

    def finish(self) -> None:
        """`finish_unhooked`, then the compute stream waits for every in-backward update
        (before the next forward reads the parameters)."""
        self.finish_unhooked()
        for device, ring in self.rings.items():
            ring.dm.current_stream(device).wait_stream(ring.opt)

    def drain(self) -> None:
        """Land every in-flight copy (call after the last micro-batch's backward)."""
        self.acc.drain()

    def zero(self) -> None:
        """End the step's sums (call after the optimiser step): the next gradient of
        each parameter replaces its sum, nothing is cleared."""
        self.acc.drain()
        for lane in self.lanes.values():
            if lane.consumed is not None:
                lane.consumed.synchronize()
                lane.consumed = None
            lane.filled = False

    def filled_sums(self) -> list[Tensor]:
        """The host sums holding this step's gradients."""
        return [self.host[p] for p in self.params if self.lanes[id(p)].filled]

    def filled_params(self) -> list[nn.Parameter]:
        """The parameters of `filled_sums`, in the same order."""
        return [p for p in self.params if self.lanes[id(p)].filled]

    def restore(self, p: nn.Parameter) -> None:
        """Put the summed gradient of one parameter back on its device (None when it
        received no gradient this step)."""
        lane = self.lanes[id(p)]
        if not lane.filled:
            p.grad = None
            return
        p.grad = self.host[p].to(p.device, p.dtype, non_blocking=True)
        lane.consumed = _event_after(p.device)
        self.stats["bytes_h2d"] += p.numel() * 4

    def to_device(self) -> None:
        """Put every summed gradient back on its device (for a device optimiser)."""
        for p in self.params:
            self.restore(p)

    @torch.no_grad()
    def store(self, p: nn.Parameter) -> None:
        """Replace the host sum of `p` with `p.grad` (copied on the current stream; the
        host waits for that stream before reading) and drop the device gradient."""
        lane = self.lanes[id(p)]
        lane.host.copy_(p.grad, non_blocking=streamed(p.device))
        lane.filled = True
        self.stats["bytes_d2h"] += p.grad.nbytes
        p.grad = None

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def close(self) -> None:
        """Stop the accumulator's worker thread."""
        self.acc.close()
