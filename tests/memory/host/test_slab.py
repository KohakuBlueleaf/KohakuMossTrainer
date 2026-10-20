"""HostSlab, model-free on the host: one cycle's buffers never overlap; once all are
back the next cycle reuses the first chunk from its start, after the writer waited on
every event handed back; a cycle that outgrew the chunks leaves one chunk of its size,
and an equal cycle after it allocates nothing."""

import pytest
import torch

from kohakumoss.offload.host import slab as hs
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GROW = 8192
SIZES = [3000, 5000, 100, 4096, 7000]


class _Writer:
    def __init__(self):
        self.waited: list = []

    def wait_event(self, ev) -> None:
        self.waited.append(ev)


def _cycle(slab, writer) -> list[torch.Tensor]:
    bufs = [slab.take(n, writer) for n in SIZES]
    for i, _ in enumerate(bufs):
        slab.give(f"ev{i}")
    return bufs


def _spans(bufs) -> list[tuple[int, int]]:
    return sorted((b.data_ptr(), b.data_ptr() + b.numel()) for b in bufs)


def check_slab(make) -> None:
    slab, writer = make(), _Writer()
    first = _cycle(slab, writer)
    for (_, end), (start, _) in zip(_spans(first), _spans(first)[1:]):
        assert end <= start, "buffers of one cycle overlap"
    assert all(b.numel() >= n for b, n in zip(first, SIZES))
    grown, cut = slab.grown, slab.cut
    assert len(slab.chunks) > 1, "the first cycle fits one chunk: no growth tested"
    second = _cycle(slab, writer)
    assert writer.waited[: len(SIZES)] == [f"ev{i}" for i in range(len(SIZES))]
    assert len(slab.chunks) == 1, len(slab.chunks)
    assert slab.nbytes == hs._up(cut, GROW), (slab.nbytes, cut)
    third = _cycle(slab, writer)
    assert slab.grown == grown + 1, (slab.grown, grown)
    assert third[0].data_ptr() == second[0].data_ptr()


class _NoWait(hs.HostSlab):
    def _rewind(self, writer) -> None:
        self.events = []
        super()._rewind(writer)


class _NoMerge(hs.HostSlab):
    def _rewind(self, writer) -> None:
        for ev in self.events:
            writer.wait_event(ev)
        self.events = []
        self.chunk = self.offset = self.cut = 0


def test_slab_reuses_its_chunks_after_the_device_read_them():
    check_slab(lambda: hs.HostSlab("cpu", False, grow=GROW))
    must_fail(lambda: check_slab(lambda: _NoWait("cpu", False, grow=GROW)), "no wait")
    must_fail(
        lambda: check_slab(lambda: _NoMerge("cpu", False, grow=GROW)),
        "the outgrown chunks kept",
    )
