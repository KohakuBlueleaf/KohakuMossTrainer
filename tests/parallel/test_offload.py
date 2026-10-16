"""Activation offload and weight streaming reproduce the dense loss and gradients."""

import pytest
import torch

from kohakumoss.offload.activation import ActivationOffload
from kohakumoss.offload.stream import WeightStream
from kohakumoss.quant.convert import quantize_model
from kohakumoss.training.memory.runners import (
    CheckpointRunner,
    OffloadRunner,
    PlainRunner,
    StreamRunner,
)
from tests._util.negative import must_fail
from tests.parallel._common import batch, loss_and_grads, make_model, max_diff

TOL = {"cpu": 1e-9, "cuda": 1e-7}


def _offload_case(device, inner):
    model = make_model(device)
    off = ActivationOffload(device, min_bytes=0)
    off.exclude(model.parameters())
    model.runner = OffloadRunner(inner, off).bind(model.layers)
    out = loss_and_grads(model, batch(device))
    return out, off


@pytest.mark.cpu
@pytest.mark.parametrize("inner", ["plain", "block"])
def test_activation_offload_cpu(inner):
    runner = PlainRunner() if inner == "plain" else CheckpointRunner("block")
    (loss, grads), off = _offload_case("cpu", runner)
    ref_model = make_model()
    ref_loss, ref = loss_and_grads(ref_model, batch())
    assert off.stats["packed"] > 0
    assert abs(loss - ref_loss) < 1e-12 and max_diff(grads, ref) < 1e-12


@pytest.mark.cuda
def test_activation_offload_cuda_matches():
    dev = "cuda"
    (loss, grads), off = _offload_case(dev, PlainRunner())
    ref_loss, ref = loss_and_grads(make_model(dev), batch(dev))
    assert off.stats["packed"] > 0
    assert abs(loss - ref_loss) < 1e-10 and max_diff(grads, ref) < TOL["cuda"]


def _all(name, lin):
    return True


def _stream_case(device, trainable, quant=None, resident=2):
    model = make_model(device)
    if quant:
        quantize_model(model.layers, quant, _all, torch.float64)
    engine = WeightStream(
        model.layers,
        device,
        resident=resident,
        trainable=trainable,
        dtype=torch.float64,
    )
    model.runner = StreamRunner(CheckpointRunner("block"), engine).bind(model.layers)
    loss, grads = loss_and_grads(model, batch(device), freeze_blocks=not trainable)
    if trainable:
        engine.flush_grads()
        for (i, key), p in engine.params.items():
            grads[f"layers.{i}.{key}.weight"] = p.grad.double().clone()
    return loss, grads, engine


def _dense(device, trainable, quant=None):
    model = make_model(device)
    if quant:
        quantize_model(model.layers, quant, _all, torch.float64)
    return loss_and_grads(model, batch(device), freeze_blocks=not trainable)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("trainable", [False, True])
def test_weight_stream_matches_dense(device, trainable):
    loss, grads, engine = _stream_case(device, trainable)
    ref_loss, ref = _dense(device, trainable)
    assert engine.stats["loads"] >= 4
    assert abs(loss - ref_loss) < 1e-10
    assert max_diff(grads, ref) < TOL[device]


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_streamed_quant_base_matches_resident_quant(device):
    loss, grads, _ = _stream_case(device, False, quant="int8")
    ref_loss, ref = _dense(device, False, quant="int8")
    assert abs(loss - ref_loss) < 1e-10 and max_diff(grads, ref) < TOL[device]


@pytest.mark.cpu
def test_stream_check_catches_stale_weights():
    ref_loss, _ = _dense("cpu", False)

    def check():
        model = make_model()
        engine = WeightStream(model.layers, "cpu", resident=2, dtype=torch.float64)
        engine.host[1] = engine.host[0]
        model.runner = StreamRunner(CheckpointRunner("block"), engine).bind(
            model.layers
        )
        loss, _ = loss_and_grads(model, batch(), freeze_blocks=True)
        assert abs(loss - ref_loss) < 1e-10

    must_fail(check, "block 1 streamed with block 0's weights")
