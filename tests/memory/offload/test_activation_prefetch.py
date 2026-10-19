"""Activation offload prefetches one block ahead in backward: unpacking blocks in
reverse, only the first block's first packet is fetched on demand, every other block
was already fetched by its successor's unpack, and every device copy equals the saved
tensor; a prefetch issued only after a miss fetches every other block on demand."""

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload, Packet
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
BLOCKS = 8
PER_BLOCK = 2
NUMEL = 1 << 16


def _miss_only(self: ActivationOffload, pk: Packet) -> torch.Tensor:
    """`_load_device` that prefetches the previous block only after a miss."""
    if pk.device_copy is None:
        self.pending.setdefault(pk.block, []).append(pk)
        self._fetch(pk.block)
        self._prefetch(self, pk.block - 1)
    self.dm.current_stream(self.device).wait_event(pk.h2d)
    pk.users -= 1
    return pk.device_copy


def on_demand(engine: ActivationOffload) -> int:
    """Packets of BLOCKS blocks stored in order, unpacked in reverse; the number of
    unpacks that found no device copy (fetched on demand)."""
    saved, packets = [], []
    for b in range(BLOCKS):
        engine.current = b
        tensors = [torch.randn(NUMEL, device="cuda") for _ in range(PER_BLOCK)]
        saved.append(tensors)
        packets.append([engine.pack(t) for t in tensors])
    misses = 0
    for b in reversed(range(BLOCKS)):
        for t, pk in zip(saved[b], packets[b]):
            misses += pk.device_copy is None
            assert torch.equal(engine.unpack(pk), t)
    torch.cuda.synchronize()
    return misses


def _one_per_step(misses: int) -> None:
    assert misses == 1, misses


def test_every_block_is_prefetched_by_its_successor(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    _one_per_step(on_demand(ActivationOffload("cuda", min_bytes=0)))
    monkeypatch.setattr(ActivationOffload, "_load_device", _miss_only)
    late = on_demand(ActivationOffload("cuda", min_bytes=0))
    assert late == BLOCKS // 2, late
    must_fail(lambda: _one_per_step(late), "a prefetch issued only after a miss")
