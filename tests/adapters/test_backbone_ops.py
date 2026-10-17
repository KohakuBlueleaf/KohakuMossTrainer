"""Adapters on a backbone with fused linear+epilogue ops bound: every attach, load,
merge and removal re-binds the backbone, so no adapter is skipped by a fused kernel."""

from functools import partial

import pytest
import torch
import torch.nn.functional as F
from _adcases import make_spec
from _adtoy import perturb

from kohakumoss.adapters import adapter_set, apply, inference, load
from kohakumoss.model import opset as opset_mod
from kohakumoss.model import random_backbone
from kohakumoss.model.opset import OpSet
from tests._util.negative import must_fail
from tests.model.conftest import make_batch

pytestmark = pytest.mark.cpu
TOL = dict(rtol=1e-4, atol=1e-5)


def _situ_from_weight(situ_glu, y, w, b):
    """A stand-in fused linear_situ: like the real kernel it reads lin.weight."""
    return situ_glu(F.linear(y, w, b))


def _backbone(fused: bool):
    model = random_backbone("tiny", seed=3).requires_grad_(False)
    ops = OpSet.resolve("cpu", torch.float32)
    if fused:
        ops.fused["linear_situ"] = partial(_situ_from_weight, ops.situ_glu)
    model.bind_ops(ops)
    model.eval()
    return model


def _out(model):
    img, text, sigma, seq, _, _ = make_batch(model.cfg)
    with torch.no_grad():
        return model(img, text, sigma, seq)


def _pair(name="lora"):
    fused, plain = _backbone(True), _backbone(False)
    a = apply(fused, make_spec(name))
    perturb(a)
    b = apply(plain, make_spec(name))
    b.load_state_dict(a.state_dict())
    return fused, plain, a


def test_apply_rebinds_fused_layers():
    fused, plain, _ = _pair()
    torch.testing.assert_close(_out(fused), _out(plain), **TOL)
    assert fused.layers[0]._situ.func is opset_mod._composed_situ


def test_apply_without_rebind_loses_the_adapter(monkeypatch):
    monkeypatch.setattr(adapter_set, "rebind_owner", lambda model: None)

    def lost():
        fused, plain, _ = _pair()
        torch.testing.assert_close(_out(fused), _out(plain), **TOL)

    must_fail(lost, "a fused linear_situ reading the base weight past the adapter")


def test_remove_and_merge_restore_the_fused_kernel():
    fused, _, aset = _pair()
    aset.remove()
    assert fused.layers[0]._situ.func is opset_mod._fused_situ
    fused2, plain2, aset2 = _pair()
    aset2.merge_into(1.0)
    assert fused2.layers[0]._situ.func is opset_mod._fused_situ
    torch.testing.assert_close(_out(fused2), _out(plain2), **TOL)


def _comfy_file(tmp_path):
    source = _backbone(False)
    aset = apply(source, make_spec("lora"))
    perturb(aset)
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy", dtype=None)
    return str(path)


@pytest.mark.parametrize("mode", ["bypass", "merge"])
def test_loaded_adapter_reaches_fused_layers(mode, tmp_path):
    path = _comfy_file(tmp_path)
    fused, plain = _backbone(True), _backbone(False)
    base = _out(fused)
    stack = load(fused, path, 1.0, mode)
    load(plain, path, 1.0, mode)
    torch.testing.assert_close(_out(fused), _out(plain), **TOL)
    stack.remove()
    assert fused.layers[0]._situ.func is opset_mod._fused_situ
    assert torch.equal(_out(fused), base)


def test_load_without_rebind_loses_the_adapter(monkeypatch, tmp_path):
    path = _comfy_file(tmp_path)
    monkeypatch.setattr(inference, "rebind_owner", lambda model: None)

    def lost():
        fused, plain = _backbone(True), _backbone(False)
        load(fused, path, 1.0, "bypass")
        load(plain, path, 1.0, "bypass")
        torch.testing.assert_close(_out(fused), _out(plain), **TOL)

    must_fail(lost, "a loaded bypass adapter skipped by the fused kernel")
