"""What KohakuMoss relies on from LyCORIS itself, checked against the forward-consistent dW.

A failure here is a LyCORIS defect (the package is used as-is by the applier).
"""

import pytest
import torch
from _adcases import build, outputs
from _adtoy import inputs
from lycoris.modules import IA3Module, get_module, make_module
from lycoris.modules.tlora import clear_timestep_mask, set_timestep_mask

pytestmark = pytest.mark.cpu
TOL = dict(rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["lora", "loha", "lokr", "lokr-both"])
def test_lycoris_merged_weight_is_forward_consistent(name):
    _, aset = build(name)
    for t in aset.targets:
        merged, _ = t.lyco.get_merged_weight(1.0)
        want = t.base_weight(torch.float32) + t.unit_fn()
        torch.testing.assert_close(merged.float(), want, **TOL)


@pytest.mark.parametrize("name", ["dora-out", "doha"])
def test_lycoris_module_forward_applies_dora_in_bypass(name):
    model, aset = build(name)
    want = outputs(model)
    for t in aset.targets:
        t.linear.forward = t.lyco.forward
    torch.testing.assert_close(outputs(model), want, rtol=2e-4, atol=2e-5)


def test_lycoris_ia3_loads_from_its_own_state_dict():
    _, aset = build("ia3")
    t = aset.targets[0]
    sd = {f"x.{k}": v for k, v in t.lyco.state_dict().items()}
    kind, params = get_module(sd, "x")
    assert kind is IA3Module
    loaded = make_module(kind, params, "x", t.linear)
    torch.testing.assert_close(loaded.weight, t.lyco.weight.detach())


def test_lycoris_oft_bypass_at_multiplier():
    model, aset = build("oft")
    aset.set_multiplier(0.5)
    want = outputs(model)
    for t in aset.targets:
        t.lyco.multiplier = 0.5
        t.linear.forward = t.lyco.forward
    torch.testing.assert_close(outputs(model), want, rtol=2e-4, atol=2e-5)


def test_lycoris_oft_rescaled_diff_at_multiplier():
    _, aset = build("oft-constraint-rescale")
    for t in aset.targets:
        got = t.lyco.get_diff_weight(0.5)[0].float()
        torch.testing.assert_close(got, t.delta(0.5)[0], **TOL)


def test_lycoris_dylora_bypass_runs():
    model, aset = build("dylora")
    for t in aset.targets:
        t.lyco.bypass_mode = True
        t.linear.forward = t.lyco.forward
    assert torch.isfinite(outputs(model)).all()


def test_lycoris_tlora_accepts_a_per_token_mask():
    model, aset = build("tlora")
    n = inputs()[0].shape[0]
    rank = aset.targets[0].lyco.lora_dim
    set_timestep_mask(torch.ones(n, rank), 0)
    try:
        full = outputs(model)
    finally:
        clear_timestep_mask(0)
    torch.testing.assert_close(full, outputs(model), **TOL)
