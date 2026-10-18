"""Activation offload: an exception inside a block's forward drops the step's packets
(their host buffers return to the pool) and restores the block index."""

from contextlib import contextmanager

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@contextmanager
def _block_without_cleanup(offload: ActivationOffload, index: int):
    """The broken variant: hooks scoped to the block, nothing undone on an exception."""
    offload.current = index
    with torch.autograd.graph.saved_tensors_hooks(offload.pack, offload.unpack):
        yield


def _forward_raises(block) -> None:
    x = torch.randn(64, 64, requires_grad=True)
    with pytest.raises(RuntimeError, match="forward failed"):
        with block(2):
            (x * x).sum()
            raise RuntimeError("forward failed")


def _released(offload: ActivationOffload) -> None:
    assert offload.pending == {}, offload.pending
    assert offload.current == 0, offload.current


def test_exception_releases_packets():
    offload = ActivationOffload("cpu", min_bytes=0)
    _forward_raises(offload.block)
    assert offload.stats["packed"] > 0
    _released(offload)
    broken = ActivationOffload("cpu", min_bytes=0)
    _forward_raises(lambda i: _block_without_cleanup(broken, i))
    must_fail(lambda: _released(broken), "packets kept after a failed forward")
