"""Host -> ring-slot block copies of a weight stream and the slot events the compute
stream orders against. See docs/internals/offload.md#weight-streaming."""

import torch
from torch import Tensor

from kohakumoss.offload.weights.streaming.device import DeviceStreams
from kohakumoss.offload.weights.streaming.events import SlotEvents, recorded
from kohakumoss.offload.weights.streaming.timing import SpanTimer
from kohakumoss_common.calls import nothing


class H2DCopier:
    """One block's host tensors -> a ring slot. On a device the copy runs on the H2D
    side stream after the slot's `free` event (and block `host_ready`, its master
    write-back) and records the slot's `ready` event; pinned host copies go directly,
    pageable ones through two alternating pinned staging sets. On the host it is a
    plain copy. The bodies of `copy`, `wait(slot)` and `mark_free(slot)` are selected
    at construction."""

    def __init__(
        self,
        dev: DeviceStreams,
        slots: int,
        blocks: int,
        host_pinned: bool,
        trainable: bool,
        stats: dict,
        timer: SpanTimer,
    ):
        device, dm, side = dev.device, dev.dm, dev.streamed
        self.device = device
        self.dm = dm
        self.trainable = trainable
        self.stats = stats
        self.timer = timer
        # two pinned staging sets for unpinned host copies, each guarded by its H2D
        self.staging: list[dict[str, Tensor]] = [{}, {}]
        self.staged: list = [None, None]
        self.stage_next = 0
        if not side:
            self._copy, self._wait, self._mark_free = (
                H2DCopier.copy_host,
                nothing,
                nothing,
            )
            return
        self.h2d = dm.Stream(device)
        events = SlotEvents(dm, device, slots)
        self.ready, self.free = events.ready, events.free
        self.host_ready = recorded(dm, device, blocks)
        self._copy = H2DCopier.copy_pinned if host_pinned else H2DCopier.copy_staged
        self._wait, self._mark_free = H2DCopier._wait_ready, H2DCopier._record_free

    def copy(self, src: dict, dst: dict, slot: int, index: int) -> None:
        """Block `index`'s host tensors `src` -> ring slot `slot` (`dst`)."""
        self._copy(self, src, dst, slot, index)

    def wait(self, slot: int) -> None:
        """The compute stream waits for slot `slot`'s copy."""
        self._wait(self, slot)

    def mark_free(self, slot: int) -> None:
        """The compute stream is done reading slot `slot`."""
        self._mark_free(self, slot)

    def copy_host(self, src: dict, dst: dict, slot: int, index: int) -> None:
        for k, t in src.items():
            dst[k].copy_(t)

    def copy_pinned(self, src: dict, dst: dict, slot: int, index: int) -> None:
        self.h2d.wait_event(self.free[slot])
        self.h2d.wait_event(self.host_ready[index])
        start = self.timer.stamp(self.h2d)
        with self.dm.stream(self.h2d):
            for k, t in src.items():
                dst[k].copy_(t, non_blocking=True)
                self.stats["bytes_h2d"] += t.nbytes
        self.timer.log(self.timer.copies, start, self.h2d)
        self.ready[slot].record(self.h2d)

    def copy_staged(self, src: dict, dst: dict, slot: int, index: int) -> None:
        """Unpinned host copies through the next of two pinned staging sets: the host
        waits only for the H2D that last read that set (two loads back) and, for a
        trainable ring, for the block's master write-back."""
        self.h2d.wait_event(self.free[slot])
        if self.trainable:
            self.host_ready[index].synchronize()
            self.stats["host_syncs"] += 1
        stage = self.stage_next
        self.stage_next ^= 1
        if self.staged[stage] is not None:
            self.staged[stage].synchronize()
            self.stats["host_syncs"] += 1
        start = self.timer.stamp(self.h2d)
        with self.dm.stream(self.h2d):
            for k, t in src.items():
                dst[k].copy_(self._stage(stage, k, t), non_blocking=True)
                self.stats["bytes_h2d"] += t.nbytes
        self.timer.log(self.timer.copies, start, self.h2d)
        self.ready[slot].record(self.h2d)
        done = self.dm.Event()
        done.record(self.h2d)
        self.staged[stage] = done

    def _stage(self, stage: int, key: str, t: Tensor) -> Tensor:
        """Staging set `stage`'s pinned buffer of `key` holding `t` (its last H2D has
        finished)."""
        bufs = self.staging[stage]
        buf = bufs.get(key)
        if buf is None:
            buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            bufs[key] = buf
        buf.copy_(t)
        return buf

    def _wait_ready(self, slot: int) -> None:
        cur = self.dm.current_stream(self.device)
        start = self.timer.stamp(cur)
        cur.wait_event(self.ready[slot])
        self.timer.log(self.timer.waits, start, cur)

    def _record_free(self, slot: int) -> None:
        self.free[slot].record(self.dm.current_stream(self.device))
