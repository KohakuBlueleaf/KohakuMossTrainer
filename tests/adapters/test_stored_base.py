"""Adapters over quantized / streamed base storage read the base through base_weight().

The stand-in storage keeps an int8 weight with per-row scales and leaves
MossLinear.weight empty, the way quantized and streamed bases do.
"""

import pytest
import torch
import torch.nn.functional as F
from _adcases import make_spec, outputs
from _adtoy import perturb, toy
from safetensors.torch import load_file
from torch import Tensor, nn

from kohakumoss.adapters import apply, load_adapters
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
TOL = dict(rtol=3e-4, atol=3e-5)
CASES = [
    "lora",
    "loha",
    "lokr",
    "oft",
    "oft-constraint-rescale",
    "boft",
    "glora",
    "ia3",
    "dylora",
    "tlora",
    "dora-out",
    "dora-in",
    "lora-norms",
]


class Int8Storage(nn.Module):
    """Stand-in BaseWeight: int8 rows with fp32 scales."""

    def __init__(self, w: Tensor):
        super().__init__()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
        self.register_buffer("q", torch.round(w / scale).to(torch.int8))
        self.register_buffer("s", scale.float())
        self.shape = w.shape

    def weight(self, dtype: torch.dtype) -> Tensor:
        return (self.q.float() * self.s).to(dtype)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        return F.linear(x, self.weight(x.dtype), bias)

    def materialize(self) -> Tensor:
        return self.weight(torch.float32)


def quantized_toy():
    model = toy()
    for m in model.modules():
        if type(m).__name__ == "MossLinear":
            m.set_base(Int8Storage(m.weight.detach()))
            m.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    return model


def dense_toy():
    """The same model with the dequantized weights as dense parameters."""
    model = toy()
    with torch.no_grad():
        for m in model.modules():
            if type(m).__name__ == "MossLinear":
                m.weight.copy_(Int8Storage(m.weight).weight(m.weight.dtype))
    return model


def _adapted(model, name, **kw):
    torch.manual_seed(5)
    aset = apply(model, make_spec(name, **kw))
    perturb(aset)
    model.eval()
    return aset


@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", CASES)
def test_stored_base_matches_dense(name, mode):
    q, d = quantized_toy(), dense_toy()
    qa = _adapted(q, name, mode=mode)
    _adapted(d, name, mode=mode)
    torch.testing.assert_close(outputs(q), outputs(d), **TOL)
    assert all(t.linear.weight.numel() == 0 for t in qa.targets)


def test_stored_base_is_really_used():
    q, d = quantized_toy(), dense_toy()
    _adapted(q, "glora")
    _adapted(d, "glora")

    def zero_storage():
        with torch.no_grad():
            for m in q.modules():
                if isinstance(m, Int8Storage):
                    m.s.zero_()
        torch.testing.assert_close(outputs(q), outputs(d), **TOL)

    must_fail(zero_storage, "outputs must depend on the stored weight")


@pytest.mark.parametrize("name", ["lora", "glora", "dora-out", "oft"])
def test_merged_export_reads_the_storage(name, tmp_path):
    q, d = quantized_toy(), dense_toy()
    qa, da = _adapted(q, name), _adapted(d, name)
    pq, pd = tmp_path / "q.safetensors", tmp_path / "d.safetensors"
    qa.export(str(pq), "merged", dtype=torch.float32)
    da.export(str(pd), "merged", dtype=torch.float32)
    sq, sd = load_file(str(pq)), load_file(str(pd))
    assert set(sq) == set(sd)
    for key in sd:
        torch.testing.assert_close(sq[key], sd[key], **TOL, msg=key)


def test_merge_into_and_full_are_refused():
    q = quantized_toy()
    aset = _adapted(q, "lora")
    with pytest.raises(RuntimeError):
        aset.merge_into()
    with pytest.raises(ValueError):
        apply(quantized_toy(), make_spec("full"))


@pytest.mark.parametrize("mode", ["bypass", "merge"])
@pytest.mark.parametrize("name", ["lora", "lokr", "glora", "dora-out", "ia3"])
def test_inference_on_stored_base(name, mode, tmp_path):
    source, aset = dense_toy(), None
    aset = _adapted(source, name)
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy", dtype=None)
    q, d = quantized_toy(), dense_toy()
    q.eval()
    d.eval()
    base = outputs(q)
    stack = load_adapters(q, [(str(path), 0.8)], mode=mode)
    load_adapters(d, [(str(path), 0.8)], mode=mode)
    torch.testing.assert_close(outputs(q), outputs(d), **TOL)
    stack.remove()
    assert torch.equal(outputs(q), base)
