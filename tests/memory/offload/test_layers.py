"""Layer streaming of a frozen encoder (mixed layer structures) reproduces its output."""

import pytest
import torch
from torch import nn

from kohakumoss.offload.layers import LayerStreamer, find_layer_list, move_except
from kohakumoss.quant.convert import quantize_torch_linears
from tests._shared.tiny import DEVICES
from tests._util.negative import must_fail


class _A(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.tanh(self.lin(x))


class _B(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up, self.down = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)
        self.register_buffer("scale", torch.full((d,), 0.5))

    def forward(self, x):
        return x + self.down(torch.relu(self.up(x))) * self.scale


class _Enc(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.embed = nn.Linear(16, d)
        self.layers = nn.ModuleList([_A(d), _A(d), _B(d), _A(d), _B(d)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("quant", [None, "int8"])
@torch.no_grad()
def test_layer_streamer_matches_resident(device, quant):
    torch.manual_seed(0)
    ref = _Enc()
    enc = _Enc()
    enc.load_state_dict(ref.state_dict())
    if quant:
        quantize_torch_linears(ref, quant, torch.float32)
        quantize_torch_linears(enc, quant, torch.float32)
    x = torch.randn(7, 16)
    want = ref.to(device)(x.to(device))
    layers = find_layer_list(enc)
    assert layers is enc.layers
    move_except(enc, layers, device)
    LayerStreamer(layers, device, resident=2)
    got = enc(x.to(device))
    assert (got - want).abs().max().item() < 1e-5
    assert all(p.device.type == "cpu" for p in enc.layers.parameters())

    def check_skipped_layer():
        assert (enc.norm(enc.embed(x.to(device))) - want).abs().max().item() < 1e-5

    must_fail(check_skipped_layer, "an encoder whose layers never ran")
