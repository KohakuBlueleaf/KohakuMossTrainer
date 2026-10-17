"""Optimiser groups: WD_RULES, Muon eligibility, names, and the LoRA+ ratio on exactly
each kind's zero-initialised factors."""

import pytest
import torch
from _adcases import make_spec
from _adtoy import toy

from kohakumoss.adapters import apply
from kohakumoss.adapters.groups import is_up
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ZERO_INIT = ("lora", "loha", "lokr", "lokr-full", "glora", "dylora")


def _fresh(name, **kw):
    torch.manual_seed(0)
    return apply(toy(), make_spec(name, **kw))


def _by_name(groups) -> dict:
    return {n: g for g in groups for n in g["names"]}


@pytest.mark.parametrize("rules", ["moss", "none", "all"])
def test_weight_decay_rules(rules):
    aset = _fresh("dora-out")
    groups = _by_name(aset.param_groups(1e-3, weight_decay=0.1, rules=rules))
    for name, g in groups.items():
        decayed = g["weight_decay"] == 0.1
        match rules:
            case "none":
                assert not decayed, name
            case "all":
                assert decayed, name
            case "moss":
                assert decayed == ("dora_scale" not in name), name
    with pytest.raises(ValueError):
        aset.param_groups(1e-3, rules="bogus")


def test_muon_takes_the_two_dimensional_factors():
    aset = _fresh("dora-out")
    groups = _by_name(aset.param_groups(1e-3, use_muon=True))
    for name, g in groups.items():
        assert g["use_muon"] == ("dora_scale" not in name), name
    assert not any(g["use_muon"] for g in aset.param_groups(1e-3))


@pytest.mark.parametrize("name", ZERO_INIT)
def test_loraplus_side_is_the_zero_initialised_factor(name):
    aset = _fresh(name)
    zero = {
        n for t in aset.targets for n, p in t.lyco.named_parameters() if not p.any()
    }
    up = {
        n
        for t in aset.targets
        for n, _ in t.lyco.named_parameters()
        if is_up(n, t.record.up_side)
    }
    assert zero and up == zero, (name, up, zero)
    groups = aset.param_groups(1.0, lr_ratio_up=4.0)
    boosted = {n.split(".", 1)[1] for g in groups if g["lr"] == 4.0 for n in g["names"]}
    assert boosted == zero


def test_loraplus_rule_is_checked_against_the_init():
    def wrong_side():
        aset = _fresh("lokr-full")
        zero = {
            n for t in aset.targets for n, p in t.lyco.named_parameters() if not p.any()
        }
        up = {
            n
            for t in aset.targets
            for n, _ in t.lyco.named_parameters()
            if is_up(n, ("lokr_w2_b",))
        }
        assert up == zero

    must_fail(wrong_side, "LoKr's LoRA+ side without the full lokr_w2")
