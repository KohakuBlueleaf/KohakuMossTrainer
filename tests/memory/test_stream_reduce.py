"""The cross-rank reduction buffer of a host-stepped weight stream, model-free (three
blocks of one linear each, on the host): `reduce_with` allocates one fp32 flat buffer
of the largest block, and every block of every step is summed in a view of it, never a
fresh buffer; the reduced sums land in the host gradients."""

import pytest
import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.offload.weights.stream import WeightStream
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SHAPES = ((8, 16), (8, 16), (12, 16))


def _blocks() -> list[nn.Module]:
    out = []
    for o, i in SHAPES:
        block = nn.Module()
        block.lin = MossLinear(i, o, bias=False)
        nn.init.normal_(block.lin.weight)
        out.append(block)
    return out


def per_block(stream: WeightStream) -> None:
    """The broken variant: a fresh flat buffer for every block of every step."""
    reducer = stream.reducer
    reduce_block = reducer.reduce_block

    def fresh(index: int, grads: dict) -> None:
        reducer.flat = torch.empty_like(reducer.flat)
        reduce_block(index, grads)

    reducer.reduce_block = fresh


def check_one_buffer(broken=None, steps: int = 2) -> None:
    stream = WeightStream(_blocks(), "cpu", slots=2, pinned=False, trainable=True)
    if broken is not None:
        broken(stream)
    seen = []
    stream.reduce_with(lambda flat, index: seen.append(flat))
    assert stream.flat.numel() == max(o * i for o, i in SHAPES)
    for step in range(steps):
        stream.start(last=True)
        for key, master in stream.params.items():
            stream.on_grad(key, torch.full(master.shape, float(step + 1)))
        for key, master in stream.params.items():
            assert torch.equal(
                stream.host_grad[key], torch.full(master.shape, step + 1.0)
            )
        stream.zero_grads()
    assert len(seen) == steps * len(SHAPES)
    ptrs = {flat.untyped_storage().data_ptr() for flat in seen}
    assert ptrs == {stream.flat.untyped_storage().data_ptr()}, len(ptrs)


def test_reduction_buffer_allocated_once():
    check_one_buffer()
    must_fail(lambda: check_one_buffer(per_block), "a buffer per block")
