"""Host-side gradient accumulation off the autograd thread (D2H into a pinned staging
ring, host add on a worker thread). See docs/internals/offload.md#host-workers."""

import threading
import time
from functools import partial

import torch
from torch import Tensor

from kohakumoss.offload.accel import module
from kohakumoss.offload.worker import HostWorker

STAGING_SLOTS = 4
RING_WAIT = "ring_wait_s"


class _Staging:
    """One host staging buffer of the ring (bytes, grown to the largest gradient it
    took) and the event its last queued add sets."""

    def __init__(self):
        self.buffer = torch.empty(0, dtype=torch.uint8)
        self.done = threading.Event()
        self.done.set()


def _accumulate(event, write, src: Tensor, dst: Tensor, done: threading.Event) -> None:
    try:
        event.synchronize()
        write(dst, src)
    finally:
        done.set()


class HostAccumulator:
    """fp32 host sums of device gradients, filled by a background worker. Gradients
    pass through a ring of `slots` pinned staging buffers shared by every key (host
    staging bytes = slots x the largest gradient, not one buffer per key). Host seconds
    the caller waited for a busy slot accumulate in `stats[RING_WAIT]` (`stats`: the
    owner's counters)."""

    def __init__(
        self, pinned: bool = True, slots: int = STAGING_SLOTS, stats: dict | None = None
    ):
        if slots < 1:
            raise ValueError(f"HostAccumulator needs slots >= 1, got {slots}")
        self.stats = {} if stats is None else stats
        self.stats.setdefault(RING_WAIT, 0.0)
        self.pinned = pinned
        self.ring = [_Staging() for _ in range(slots)]
        self.next = 0
        self.last: dict[int, threading.Event] = {}
        self.bytes_d2h = 0
        self.worker = HostWorker("kohakumoss-grad-accumulate")

    def close(self) -> None:
        """Stop the worker thread."""
        self.worker.close()

    def staging_bytes(self) -> int:
        """Host bytes the staging ring holds."""
        return sum(s.buffer.numel() for s in self.ring)

    def _take(self, grad: Tensor) -> tuple[Tensor, threading.Event]:
        """The next staging buffer as `grad`'s shape / dtype, once its last add landed."""
        slot = self.ring[self.next]
        self.next = (self.next + 1) % len(self.ring)
        if not slot.done.is_set():
            start = time.perf_counter()
            slot.done.wait()
            self.stats[RING_WAIT] += time.perf_counter() - start
        self._raise()
        nbytes = grad.nbytes
        if slot.buffer.numel() < nbytes:
            slot.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self.pinned)
        slot.done.clear()
        view = slot.buffer[:nbytes].view(grad.dtype).view(grad.shape)
        return view, slot.done

    def add(
        self, key: int, grad: Tensor, dst: Tensor, stream, first: bool = False
    ) -> None:
        """Queue `dst += grad` (device grad, host fp32 dst; `first`: `dst = grad`, so
        a sum never needs zeroing) without blocking backward."""
        buf, done = self._take(grad)
        dm = module(grad.device)
        stream.wait_stream(dm.current_stream(grad.device))
        with dm.stream(stream):
            buf.copy_(grad, non_blocking=True)
        grad.record_stream(stream)
        ev = dm.Event()
        ev.record(stream)
        self.bytes_d2h += grad.nbytes
        write = Tensor.copy_ if first else Tensor.add_
        self.last[key] = done
        self.worker.put(partial(_accumulate, ev, write, buf, dst, done))

    def wait(self, key: int) -> None:
        """Every queued add for `key` has landed (the worker runs adds in order)."""
        done = self.last.get(key)
        if done is not None:
            done.wait()
        self._raise()

    def drain(self) -> None:
        """Every queued add has landed."""
        self.worker.join()

    def _raise(self) -> None:
        if self.worker.errors:
            raise self.worker.errors.pop(0)
