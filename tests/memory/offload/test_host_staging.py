"""HostAccumulator stages gradients through a shared ring: its host staging is `slots`
x the largest gradient, whatever the number of keys, and a slot is reused only after
its queued add has landed (model-free, CPU tensors)."""

import threading

import pytest
import torch

from kohakumoss.offload.optim import host_acc
from kohakumoss.offload.optim.host_acc import HostAccumulator
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SLOTS = 4
SHAPES = [(64, 32), (128,), (16, 16, 4), (512,), (8,)] * 6


def _stage_all(acc: HostAccumulator) -> list:
    views = []
    for shape in SHAPES:
        view, done = acc._take(torch.empty(shape, dtype=torch.bfloat16))
        assert view.shape == shape and view.dtype == torch.bfloat16
        views.append(view)
        done.set()
    return views


def check_bounded(acc: HostAccumulator) -> None:
    _stage_all(acc)
    largest = max(torch.Size(s).numel() for s in SHAPES) * 2
    assert acc.staging_bytes() <= SLOTS * largest, acc.staging_bytes()


class _PerKey(HostAccumulator):
    """A staging buffer per gradient, kept forever."""

    def _take(self, grad):
        slot = host_acc._Staging()
        slot.buffer = torch.empty(grad.numel() * grad.element_size(), dtype=torch.uint8)
        self.ring.append(slot)
        slot.done.clear()
        return slot.buffer.view(grad.dtype).view(grad.shape), slot.done


def test_staging_is_slots_times_the_largest():
    acc = HostAccumulator(pinned=False, slots=SLOTS)
    try:
        check_bounded(acc)
    finally:
        acc.close()
    broken = _PerKey(pinned=False, slots=SLOTS)
    try:
        must_fail(lambda: check_bounded(broken), "one staging buffer per gradient")
    finally:
        broken.close()


def check_waits(acc: HostAccumulator, timeout: float) -> None:
    grad = torch.empty(16)
    _, done = acc._take(grad)
    took = threading.Event()
    thread = threading.Thread(target=lambda: (acc._take(grad), took.set()))
    thread.start()
    assert not took.wait(timeout), "a slot was reused before its add landed"
    done.set()
    assert took.wait(5.0)
    thread.join()


class _NoWait(HostAccumulator):
    """Reuses a slot without waiting for its last add."""

    def _take(self, grad):
        for slot in self.ring:
            slot.done.set()
        return super()._take(grad)


def test_a_slot_waits_for_its_add():
    acc = HostAccumulator(pinned=False, slots=1)
    try:
        check_waits(acc, 0.2)
    finally:
        acc.close()
    broken = _NoWait(pinned=False, slots=1)
    try:
        must_fail(lambda: check_waits(broken, 0.2), "slot reused while in flight")
    finally:
        broken.close()


def check_wait_counted(acc: HostAccumulator, held: float) -> None:
    grad = torch.empty(16)
    acc._take(grad)[1].set()
    assert acc.stats[host_acc.RING_WAIT] == 0.0, "a free slot counted as a wait"
    _, done = acc._take(grad)
    thread = threading.Thread(target=lambda: acc._take(grad))
    thread.start()
    threading.Timer(held, done.set).start()
    thread.join(5.0)
    assert acc.stats[host_acc.RING_WAIT] >= held * 0.5, acc.stats


class _Uncounted(HostAccumulator):
    """Waits for a busy slot without counting the wait."""

    def _take(self, grad):
        self.ring[self.next].done.wait()
        return super()._take(grad)


def test_ring_wait_is_counted_in_the_owner_stats():
    stats = {"bytes_d2h": 0}
    acc = HostAccumulator(pinned=False, slots=1, stats=stats)
    try:
        check_wait_counted(acc, 0.2)
        assert acc.stats is stats
    finally:
        acc.close()
    broken = _Uncounted(pinned=False, slots=1)
    try:
        must_fail(lambda: check_wait_counted(broken, 0.2), "ring wait not counted")
    finally:
        broken.close()


def test_slots_must_be_positive():
    with pytest.raises(ValueError):
        HostAccumulator(pinned=False, slots=0)
