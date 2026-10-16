"""Host-side gradient accumulation off the autograd thread.

A device gradient is copied into a pinned staging buffer on a D2H stream; a worker
thread waits for the copy and adds it into the fp32 host accumulator, so backward never
blocks on a host add. Each accumulator alternates between two staging buffers.
"""

import queue
import threading

import torch
from torch import Tensor


class _Slot:
    def __init__(self, shape, dtype, pinned: bool):
        self.buffers = [
            torch.empty(shape, dtype=dtype, pin_memory=pinned) for _ in range(2)
        ]
        self.done = [threading.Event(), threading.Event()]
        for ev in self.done:
            ev.set()
        self.next = 0


class HostAccumulator:
    """fp32 host sums of device gradients, filled by a background worker."""

    def __init__(self, pinned: bool = True):
        self.pinned = pinned
        self.jobs: queue.Queue = queue.Queue()
        self.slots: dict[int, _Slot] = {}
        self.errors: list[BaseException] = []
        self.bytes_d2h = 0
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                event, src, dst, done = job
                event.synchronize()
                dst.add_(src)
            except BaseException as err:
                self.errors.append(err)
            finally:
                done.set()
                self.jobs.task_done()

    def add(self, key: int, grad: Tensor, dst: Tensor, stream) -> None:
        """Queue `dst += grad` (device grad, host fp32 dst) without blocking backward."""
        slot = self.slots.get(key)
        if slot is None:
            slot = _Slot(grad.shape, grad.dtype, self.pinned)
            self.slots[key] = slot
        b = slot.next
        slot.next ^= 1
        slot.done[b].wait()
        slot.done[b].clear()
        buf = slot.buffers[b]
        stream.wait_stream(torch.cuda.current_stream(grad.device))
        with torch.cuda.stream(stream):
            buf.copy_(grad, non_blocking=True)
        grad.record_stream(stream)
        ev = torch.cuda.Event()
        ev.record(stream)
        self.bytes_d2h += grad.numel() * grad.element_size()
        self.jobs.put((ev, buf, dst, slot.done[b]))

    def wait(self, key: int) -> None:
        """Every queued add for `key` has landed."""
        slot = self.slots.get(key)
        if slot is not None:
            for ev in slot.done:
                ev.wait()
        self._raise()

    def drain(self) -> None:
        """Every queued add has landed."""
        self.jobs.join()
        self._raise()

    def _raise(self) -> None:
        if self.errors:
            raise self.errors.pop(0)
