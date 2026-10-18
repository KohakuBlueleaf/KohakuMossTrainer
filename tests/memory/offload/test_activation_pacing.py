"""Activation offload paces the host: with the device-to-host stream lagging, once a
packet of block `b` is stored every copy of the blocks before `b - inflight + 1` has run,
so their device tensors are free again; without the pacing they are still queued."""

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
LAG = 20_000_000
NUMEL = 1 << 18
BLOCKS = 6
INFLIGHT = 2


def _store_lagged(engine: ActivationOffload) -> list[list]:
    """Two packets per block, the copy stream lagged before each block; after each
    block the copy events of every earlier block."""
    seen, done = [], []
    for b in range(BLOCKS):
        engine.current = b
        with torch.cuda.stream(engine.d2h_stream):
            torch.cuda._sleep(LAG)
        events = [engine._store_device(torch.randn(NUMEL, device="cuda")).d2h]
        events.append(engine._store_device(torch.randn(NUMEL, device="cuda")).d2h)
        seen.append(events)
        done.append([[ev.query() for ev in evs] for evs in seen[: b + 1]])
    torch.cuda.synchronize()
    return done


def check_paced(done: list[list]) -> None:
    for b, blocks in enumerate(done):
        for older, flags in enumerate(blocks[: max(b - INFLIGHT + 1, 0)]):
            assert all(flags), (b, older, flags)


def test_offload_copies_stay_within_the_inflight_blocks(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    engine = ActivationOffload("cuda", min_bytes=0, inflight=INFLIGHT)
    done = _store_lagged(engine)
    check_paced(done)
    assert engine.stats["paced"] == BLOCKS - INFLIGHT
    monkeypatch.setattr(ActivationOffload, "_pace", lambda self, kind, block, ev: None)
    unpaced = _store_lagged(ActivationOffload("cuda", min_bytes=0, inflight=INFLIGHT))
    must_fail(
        lambda: check_paced(unpaced), "an offload that never waits for its copies"
    )
