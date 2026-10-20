"""The host text-encoder placement, model-free on constructed linears: decoder layers
whose host form holds another dtype (the bf16 file pages of an fp16 run) keep that form
between passes and are cast per layer during one, so no whole cast copy is held; the
small rest is cast once, the embedding rows at lookup; layers already in the compute
dtype need no ring."""

import pytest
import torch
from torch import nn

from kohakumoss.offload.encoders.text import HostPlacement, TextPlacement
from kohakumoss.offload.weights.layers import module_tensors
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DIM = 16
LAYERS = 3


def _model() -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleList(
        nn.Linear(DIM, DIM, bias=False, dtype=torch.bfloat16) for _ in range(LAYERS)
    )
    model.norm = nn.LayerNorm(DIM, dtype=torch.bfloat16)
    model.embed_tokens = nn.Embedding(10, DIM, dtype=torch.bfloat16)
    return model


def check_host(make) -> None:
    model = _model()
    dtypes = {id(t): torch.float16 for t in module_tensors(model).values()}
    want = [layer.weight.detach().to(torch.float16) for layer in model.layers]
    held = [layer.weight.data_ptr() for layer in model.layers]
    place = make(model, model.layers, model.embed_tokens, "cpu", False, dtypes=dtypes)
    assert [layer.weight.data_ptr() for layer in model.layers] == held
    place.begin()
    for i, layer in enumerate(model.layers):
        place.enter[i]()
        assert layer.weight.dtype == torch.float16
        assert torch.equal(layer.weight, want[i])
        place.leave[i]()
    place.end()
    assert [layer.weight.data_ptr() for layer in model.layers] == held
    assert model.norm.weight.dtype == torch.float16
    rows = place.embed(torch.tensor([1, 2]))
    assert rows.dtype == torch.float16, rows.dtype
    assert model.embed_tokens.weight.dtype == torch.bfloat16


def _cast_all(model, layers, embedding, device, pinned, dtypes):
    return TextPlacement(model, layers, embedding, device, pinned, dtypes=dtypes)


def test_host_layers_are_cast_per_pass():
    check_host(HostPlacement)
    must_fail(lambda: check_host(_cast_all), "every layer held as a cast copy")
    model = _model()
    plain = HostPlacement(model, model.layers, model.embed_tokens, "cpu", False)
    assert plain.streamer is None
