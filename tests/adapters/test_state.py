"""Raw state round trips bit-exactly; parameter groups follow the decay and LoRA+ rules."""

import pytest
import torch
from _adcases import CASES, build, make_spec, outputs
from _adtoy import toy
from safetensors.torch import load_file

from kohakumoss.adapters import apply
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("name", sorted(CASES))
def test_state_round_trip_is_exact(name, tmp_path):
    model, aset = build(name)
    want = outputs(model)
    path = tmp_path / "state.safetensors"
    aset.save_state(str(path))
    fresh = toy()
    other = apply(fresh, make_spec(name))
    fresh.eval()

    def mismatched():
        assert torch.equal(outputs(fresh), want)

    must_fail(mismatched, "a freshly initialised adapter differs")
    other.load_state_dict(load_file(str(path)))
    assert torch.equal(outputs(fresh), want)


def test_unknown_state_key_is_an_error():
    _, aset = build("lora")
    with pytest.raises(KeyError):
        aset.load_state_dict({"nope.weight": torch.zeros(1)})


def test_param_groups_decay_and_ratio():
    _, aset = build("dora-out", lr_ratio_up=4.0)
    groups = aset.param_groups(1e-3, weight_decay=0.1)
    params = [p for g in groups for p in g["params"]]
    assert len(params) == len(list(aset.trainable_params()))
    by_id = {id(p): g for g in groups for p in g["params"]}
    t = aset.targets[0]
    assert by_id[id(t.lyco.lora_up.weight)]["lr"] == pytest.approx(4e-3)
    assert by_id[id(t.lyco.lora_down.weight)]["lr"] == pytest.approx(1e-3)
    assert by_id[id(t.lyco.dora_scale)]["weight_decay"] == 0.0
    assert by_id[id(t.lyco.lora_down.weight)]["weight_decay"] == 0.1


def test_merge_into_bakes_and_detaches():
    model, aset = build("lokr")
    want = outputs(model)
    aset.merge_into()
    assert all("forward" not in vars(m) for m in model.modules())
    assert not any(".lyco." in k for k in model.state_dict())
    torch.testing.assert_close(outputs(model), want, rtol=2e-4, atol=2e-5)


def test_remove_restores_base_forward():
    model, aset = build("full")
    aset.remove()
    plain = toy()
    plain.eval()
    torch.testing.assert_close(outputs(model), outputs(plain))
