"""Bound forwards equal independent dW formulas; bypass equals merge; DoRA is applied."""

from functools import partialmethod

import pytest
import torch
from _adcases import CASES, build, outputs, reference_model
from _adref import lycoris_delta
from _adtoy import inputs, toy
from lycoris.modules import DyLoraModule

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
TOL = dict(rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("name", sorted(CASES))
def test_forward_matches_independent_reference(name):
    model, aset = build(name)
    torch.testing.assert_close(outputs(model), outputs(reference_model(aset)), **TOL)


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n in CASES
        if CASES[n].get("mode") != "merge" and CASES[n]["algo"] != "tlora"
    ),
)
def test_bypass_equals_merge(name):
    bypass, _ = build(name)
    merged, _ = build(name, mode="merge")
    torch.testing.assert_close(outputs(bypass), outputs(merged), **TOL)


@pytest.mark.parametrize("name", ["lora", "loha", "lokr", "oft", "boft", "dora-out"])
def test_multiplier_zero_is_the_base_model(name):
    model, aset = build(name)
    full = outputs(model)
    aset.set_multiplier(0.0)
    zero = outputs(model)
    assert not torch.allclose(full, zero)
    plain = toy()
    plain.eval()
    torch.testing.assert_close(zero, outputs(plain), **TOL)


@pytest.mark.parametrize("name", ["lora", "lokr", "dora-out", "dora-in", "glora"])
def test_delta_matches_reference(name):
    _, aset = build(name)
    native = aset.tensors("lycoris")
    for t in aset.targets:
        parts = {
            k[len(t.lora_name) + 1 :]: v
            for k, v in native.items()
            if k.startswith(t.lora_name + ".")
        }
        w = t.base_weight(torch.float64)
        algo = {"locon": "locon", "lokr": "lokr", "glora": "glora"}[t.kind]
        ref = lycoris_delta(algo, parts, w, w.shape[0])
        torch.testing.assert_close(t.delta()[0].double(), ref, rtol=1e-4, atol=1e-6)


def test_dora_is_applied_in_bypass():
    model, aset = build("dora-out")
    ref = outputs(reference_model(aset))
    torch.testing.assert_close(outputs(model), ref, **TOL)

    def dora_dropped():
        for t in aset.targets:
            t.wd = False
        aset.bind()
        torch.testing.assert_close(outputs(model), ref, **TOL)

    must_fail(dora_dropped, "a bypass that skips the DoRA epilogue")


@pytest.mark.parametrize("name", ["dora-out", "dora-in", "doha"])
def test_dora_multiplier_interpolates_to_base(name):
    model, aset = build(name)
    aset.set_multiplier(0.0)
    plain = toy()
    plain.eval()
    torch.testing.assert_close(outputs(model), outputs(plain), **TOL)


def test_reference_catches_wrong_scale():
    model, aset = build("loha")
    ref = reference_model(aset)

    def doubled():
        with torch.no_grad():
            for t in aset.targets:
                t.lyco.hada_w2_a.mul_(2)
        aset.bind()
        torch.testing.assert_close(outputs(model), outputs(ref), **TOL)

    must_fail(doubled, "reference must notice a factor change")


def _doubled_get_weight(self, real, rank):
    down, up, scale = real(self, rank)
    return down, up, scale * 2


@pytest.mark.parametrize("name", ["tlora", "dylora"])
def test_reference_is_independent_of_the_module_scale(name, monkeypatch):
    model, aset = build(name)

    def slipped():
        if name == "dylora":
            real = DyLoraModule.get_weight
            monkeypatch.setattr(
                DyLoraModule, "get_weight", partialmethod(_doubled_get_weight, real)
            )
        for t in aset.targets:
            t.lyco.scale = t.lyco.scale * 2
        ref = reference_model(aset)
        aset.bind()
        torch.testing.assert_close(outputs(model), outputs(ref), **TOL)

    must_fail(slipped, f"a {name} scale slip shared by forward and export")


@pytest.mark.parametrize("name", ["lora", "lokr", "oft", "full"])
def test_backward_reaches_every_adapter_tensor(name):
    model, aset = build(name)
    model.train()
    latent, text = inputs()
    model(latent, text).square().sum().backward()
    for key, p in aset.named_trainable():
        assert p.grad is not None and torch.isfinite(p.grad).all(), key
