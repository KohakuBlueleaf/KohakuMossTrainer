"""The host forms of the frozen-encoder offload, model-free: `host="file"` keeps each
host tensor's own storage (no copy) and casts to its device dtype (`dtypes`) on the
way in; `host="copy"` holds copies already in the device dtype. Both give the values
of the cast model, for HostCopies and a LayerStreamer ring."""

import pytest
import torch
from torch import nn

from kohakumoss.offload.encoders.placement import HostCopies
from kohakumoss.offload.weights.layers import LayerStreamer, module_tensors
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DIM = 8
STORED, COMPUTE = torch.bfloat16, torch.float16


def _layers() -> nn.ModuleList:
    """Three linears stored in bf16 (the checkpoint's dtype)."""
    torch.manual_seed(0)
    return nn.ModuleList(nn.Linear(DIM, DIM, dtype=STORED) for _ in range(3))


def _dtypes(layers) -> dict[int, torch.dtype]:
    return {id(t): COMPUTE for m in layers for t in module_tensors(m).values()}


def assert_copies(host: str, make=HostCopies) -> None:
    layers = _layers()
    tensors = [t for m in layers for t in module_tensors(m).values()]
    want = [t.detach().to(COMPUTE) for t in tensors]
    ptrs = [t.data_ptr() for t in tensors]
    copies = make(tensors, torch.device("cpu"), False, host, _dtypes(layers))
    held = [t.data_ptr() for t in tensors]
    assert (held == ptrs) == (host == "file")
    if host == "copy":
        assert all(t.dtype == COMPUTE for t in tensors)
    copies.begin()
    assert all(t.dtype == COMPUTE and torch.equal(t, w) for t, w in zip(tensors, want))
    copies.end()
    assert [t.data_ptr() for t in tensors] == held


@pytest.mark.parametrize("host", ["copy", "file"])
def test_host_copies_forms(host):
    assert_copies(host)


def assert_ring(host: str, make=LayerStreamer) -> None:
    layers = _layers()
    ref = [nn.Linear(DIM, DIM, dtype=COMPUTE) for _ in layers]
    for r, m in zip(ref, layers):
        r.load_state_dict({k: v.to(COMPUTE) for k, v in m.state_dict().items()})
    ptrs = [m.weight.data_ptr() for m in layers]
    x = torch.randn(2, DIM, dtype=COMPUTE)
    with torch.no_grad():
        want = x
        for r in ref:
            want = r(want)
        streamer = make(layers, "cpu", 2, False, True, host, _dtypes(layers))
        got = x
        for m in layers:
            got = m(got)
    streamer.close()
    assert torch.equal(got, want)
    kept = [m.weight.data_ptr() for m in layers] == ptrs
    assert kept == (host == "file")


@pytest.mark.parametrize("host", ["copy", "file"])
def test_streamer_forms(host):
    assert_ring(host)


class _NoCast(HostCopies):
    def begin(self) -> None:
        for t, h, _ in self.pairs:
            t.data = h.to(self.device)


def _streamer_copying(layers, device, slots, pinned, hooks, host, dtypes):
    return LayerStreamer(layers, device, slots, pinned, hooks, "copy", dtypes)


def test_checks_fail_on_broken_forms():
    must_fail(lambda: assert_copies("file", _NoCast), "file pages not cast")
    must_fail(lambda: assert_ring("file", _streamer_copying), "file form copied")
