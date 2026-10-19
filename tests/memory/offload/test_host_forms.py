"""The host forms of the frozen-encoder offload, model-free: `host="file"` keeps each
host tensor's own storage (no copy) and casts to its device dtype (`dtypes`) on the
way in; `host="copy"` holds copies already in the device dtype; a Deferred host form
moves its int8 base and builds the dense weight on the way in. All give the values
of the cast (dequantized) model, for HostCopies and a LayerStreamer ring."""

import pytest
import torch
from torch import nn

from kohakumoss.offload.encoders.placement import HostCopies
from kohakumoss.offload.weights.deferred import Deferred, placeholder
from kohakumoss.offload.weights.layers import LayerStreamer, module_tensors
from kohakumoss.quant.weights import QuantWeight
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


def _quantized_layers():
    """Three linears whose weights are Deferred over an int8 base (placeholders between
    passes), the dense weights they stand for, and the Deferred map."""
    layers = _layers()
    dense, deferred = [], {}
    for m in layers:
        base = QuantWeight.from_weight(m.weight.detach().float(), "int8")
        dense.append(base.weight(COMPUTE))
        m.weight = nn.Parameter(placeholder(m.weight.shape, COMPUTE), False)
        m.bias = nn.Parameter(m.bias.detach().to(COMPUTE), False)
        deferred[id(m.weight)] = Deferred(base, m.weight.shape)
    return layers, dense, deferred


def assert_deferred_copies(make=HostCopies) -> None:
    layers, dense, deferred = _quantized_layers()
    weights = [m.weight for m in layers]
    copies = make(weights, torch.device("cpu"), False, "copy", None, deferred)
    assert copies.nbytes == sum(d.nbytes for d in deferred.values())
    copies.begin()
    assert all(torch.equal(w, d) for w, d in zip(weights, dense))
    copies.end()
    assert all(w.untyped_storage().nbytes() == 2 for w in weights)


def test_deferred_host_form_builds_on_the_way_in():
    assert_deferred_copies()


def test_deferred_ring_runs_the_dense_layers():
    layers, dense, deferred = _quantized_layers()
    x = torch.randn(2, DIM, dtype=COMPUTE)
    with torch.no_grad():
        want = x
        for m, w in zip(layers, dense):
            want = torch.nn.functional.linear(want, w, m.bias)
        streamer = LayerStreamer(layers, "cpu", 2, False, True, "file", None, deferred)
        got = x
        for m in layers:
            got = m(got)
    streamer.close()
    assert torch.equal(got, want)
    assert streamer.h2d_bytes == sum(streamer.layer_bytes)


class _NoBuild(HostCopies):
    def begin(self) -> None:
        for t, h, _ in self.pairs:
            t.data = h.to(self.device)


class _NoCast(HostCopies):
    def begin(self) -> None:
        for t, h, _ in self.pairs:
            t.data = h.to(self.device)


def _streamer_copying(layers, device, slots, pinned, hooks, host, dtypes):
    return LayerStreamer(layers, device, slots, pinned, hooks, "copy", dtypes)


def test_checks_fail_on_broken_forms():
    must_fail(lambda: assert_copies("file", _NoCast), "file pages not cast")
    must_fail(lambda: assert_ring("file", _streamer_copying), "file form copied")
    must_fail(lambda: assert_deferred_copies(_NoBuild), "placeholders moved as is")
