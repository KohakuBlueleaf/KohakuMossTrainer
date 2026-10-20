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
WIDEN_CHUNK = 1 << 22  # elements per fp32 chunk of a widening add


class _Staging:
    """One host staging buffer of the ring (bytes, grown to the largest gradient it
    took) and the event its last queued add sets."""

    def __init__(self):
        self.buffer = torch.empty(0, dtype=torch.uint8)
        self.done = threading.Event()
        self.done.set()


def widening_add(dst: Tensor, src: Tensor, scratch: Tensor) -> None:
    """`dst += src` for a 16-bit `src` and an fp32 `dst` (same shape), through
    `scratch` (fp32, WIDEN_CHUNK elements) one chunk at a time; bitwise the same sum as
    `dst.add_(src)`. See docs/internals/offload.md#host-workers."""
    flat_dst, flat_src = dst.view(-1), src.reshape(-1)
    n = flat_dst.numel()
    for start in range(0, n, WIDEN_CHUNK):
        stop = min(start + WIDEN_CHUNK, n)
        part = scratch[: stop - start]
        part.copy_(flat_src[start:stop])
        flat_dst[start:stop].add_(part)


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
        self.scratch = torch.empty(WIDEN_CHUNK, dtype=torch.float32)
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
        write = Tensor.copy_ if first else self._adder(grad.dtype, dst.dtype)
        self.last[key] = done
        self.worker.put(partial(_accumulate, ev, write, buf, dst, done))

    def _adder(self, src: torch.dtype, dst: torch.dtype):
        """`dst += src` for these dtypes: a widening add through the worker's fp32
        scratch for a 16-bit gradient into an fp32 sum, else `Tensor.add_`."""
        if src == dst or dst != torch.float32:
            return Tensor.add_
        return partial(widening_add, scratch=self.scratch)

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
