"""Layer streaming and activation offload are freed by reference counting alone: a
LayerStreamer with forward hooks on its layers, a ring text-encoder placement and an
ActivationOffload engine are dead (with their layers' weights) right after `del`, with
the cyclic collector off; each with one strong self-reference put back stays alive."""

import types
from functools import partial

import pytest
import torch
from torch import nn

from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.offload.encoders.text import RingPlacement
from kohakumoss.offload.weights.layers import LayerStreamer
from tests._util.negative import must_fail
from tests._util.release import check_released

pytestmark = pytest.mark.cpu
DIM = 8
LAYERS = 3


def _layers() -> nn.ModuleList:
    return nn.ModuleList(nn.Linear(DIM, DIM) for _ in range(LAYERS))


def _through(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        x = layer(x)
    return x


def _streamer(holds_layers: bool = False) -> tuple:
    """A hooked streamer over three linears; one forward through the ring equals the
    plain forward."""
    layers = _layers()
    x = torch.randn(2, DIM)
    with torch.no_grad():
        want = _through(layers, x)
        streamer = LayerStreamer(layers, "cpu", slots=2, pinned=False)
        if holds_layers:
            streamer.layers = list(layers)
        got = _through(layers, x)
    streamer.close()
    assert torch.equal(got, want)
    return streamer, layers[0].weight


def test_layer_streamer_is_freed_on_del():
    check_released(_streamer)
    must_fail(
        lambda: check_released(partial(_streamer, holds_layers=True)),
        "a streamer holding the layers its hooks refer to it from",
    )


def _encoder() -> nn.Module:
    model = nn.Module()
    model.layers = _layers()
    model.norm = nn.LayerNorm(DIM)
    return model


def _ring(strong_enter: bool = False) -> tuple:
    """A ring placement driven by its enter / leave lists for one pass."""
    model = _encoder()
    place = RingPlacement(model, model.layers, None, "cpu", pinned=False, slots=2)
    if strong_enter:
        place.enter = [lambda i=i: place.streamer.enter(i) for i in range(LAYERS)]
    place.begin()
    for i in range(LAYERS):
        place.enter[i]()
        place.leave[i]()
    place.end()
    return place, model.layers[0].weight


def test_ring_placement_is_freed_on_del():
    check_released(_ring)
    must_fail(
        lambda: check_released(partial(_ring, strong_enter=True)),
        "enter lambdas over the placement",
    )


def _offload(strong_load: bool = False) -> tuple:
    """One block's saved tensors packed and unpacked by the host path; the gradient
    is the plain one."""
    off = ActivationOffload("cpu", min_bytes=0)
    if strong_load:
        off._load = types.MethodType(ActivationOffload._load_host, off)
    w = torch.randn(64, requires_grad=True)
    x = torch.randn(64)
    with off.block(0):
        y = (w * x).sin().sum()
    if not strong_load:
        y.backward()
        assert torch.allclose(w.grad, x * (w * x).cos().detach())
    assert off.stats["packed"] > 0
    return (off,)


def test_activation_offload_is_freed_on_del():
    check_released(_offload)
    must_fail(
        lambda: check_released(partial(_offload, strong_load=True)),
        "a bound load method on the engine",
    )
