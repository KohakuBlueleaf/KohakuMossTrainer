"""Exports and merges are deterministic in any mode; merged weights round once; state
loads are exact; merge_into bakes every delta at its argument."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from _adcases import build, make_spec, outputs
from _adtoy import perturb
from safetensors.torch import load_file
from test_export import backbone_set
from test_stored_base import tiny_pair

from kohakumoss.adapters import adapter_set, apply
from kohakumoss.adapters.bind import fused
from kohakumoss.adapters.norms import NormDelta
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DROPPED = ("lora", "dora-out", "loha")
ROWS = "layers.0.proj.q"


def _rows(aset):
    t = next(t for t in aset.targets if t.path == ROWS)
    return {ROWS: torch.linspace(0.5, 1.5, t.lyco.shape[0])}


def _exports(aset, tmp_path, tag):
    aset.export(str(tmp_path / f"{tag}m.safetensors"), "merged", dtype=torch.float32)
    merged = load_file(str(tmp_path / f"{tag}m.safetensors"))
    comfy = aset.tensors("comfy", row_scales=_rows(aset))
    return merged, comfy


def _same(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


@pytest.mark.parametrize("name", DROPPED)
def test_train_mode_exports_equal_eval_exports(name, tmp_path):
    model, aset = backbone_set(name, rank_dropout=0.5)
    model.train()
    first, comfy1 = _exports(aset, tmp_path, "a")
    second, comfy2 = _exports(aset, tmp_path, "b")
    model.eval()
    ref, comfy_ref = _exports(aset, tmp_path, "c")
    _same(first, ref)
    _same(second, ref)
    _same(comfy1, comfy_ref)
    _same(comfy2, comfy_ref)


def test_train_mode_export_without_eval_scope_is_caught(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter_set, "evaluated", lambda targets: nullcontext())

    def drawn():
        model, aset = backbone_set("lora", rank_dropout=0.5)
        model.train()
        first, _ = _exports(aset, tmp_path, "a")
        second, _ = _exports(aset, tmp_path, "b")
        _same(first, second)

    must_fail(drawn, "an export taken with rank dropout drawing")


def test_merge_into_bakes_norms_at_the_argument():
    model, aset = build("lora-norms")
    aset.set_multiplier(0.7)
    norm = aset.norms[0]
    w0 = norm.module.parametrizations["weight"].original.detach().clone()
    d = norm.weight_delta.w_norm.detach().clone()
    aset.merge_into(1.0)
    torch.testing.assert_close(norm.module.weight, (w0 + d).to(w0.dtype))


def test_load_state_dict_rejects_a_partial_state():
    _, aset = build("lora")
    state = aset.state_dict()
    state.pop(next(iter(state)))
    with pytest.raises(KeyError, match="missing"):
        aset.load_state_dict(state)
    with pytest.raises(KeyError, match="unknown"):
        aset.load_state_dict({**aset.state_dict(), "nope.weight": torch.zeros(1)})


def test_norm_delta_rounds_once():
    original = torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16)
    nd = NormDelta(3)
    with torch.no_grad():
        nd.w_norm.copy_(torch.tensor([0.0039063, 0.0, -0.0019532]))
    want = (original.float() + nd.w_norm).to(torch.bfloat16)
    assert torch.equal(nd(original), want)
    twice = original + nd.w_norm.to(torch.bfloat16)
    assert not torch.equal(twice, want)


def test_fused_merge_forward_rounds_w_plus_dw_once():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(16, 16, generator=g)
    dw = torch.randn(16, 16, generator=g) * 1e-3
    x = torch.randn(4, 16, generator=g).to(torch.bfloat16)

    def base_fn(dtype):
        return w.to(dtype)

    def unit(base):
        return dw

    got = fused.merged_forward(base_fn, unit, fused.same, fused.lin_nobias, x)
    assert torch.equal(got, F.linear(x, (w + dw).to(torch.bfloat16)))
    twice = F.linear(x, w.to(torch.bfloat16) + dw.to(torch.bfloat16))
    assert not torch.equal(twice, got)


def test_stored_base_diffusers_export(tmp_path):
    q, d = tiny_pair()
    torch.manual_seed(5)
    qa = apply(q, make_spec("lora"))
    perturb(qa)
    da = apply(d, make_spec("lora"))
    da.load_state_dict(qa.state_dict())
    for model, aset, tag in ((q, qa, "q"), (d, da, "d")):
        aset.export(str(tmp_path / tag), "diffusers", dtype=torch.float32)
    sq = {}
    sd = {}
    for tag, out in (("q", sq), ("d", sd)):
        for f in (tmp_path / tag).glob("*.safetensors"):
            out.update(load_file(str(f)))
    assert set(sq) == set(sd)
    for k in sd:
        torch.testing.assert_close(sq[k], sd[k], rtol=3e-4, atol=3e-5, msg=k)


def test_outputs_unchanged_by_export():
    model, aset = build("lora", rank_dropout=0.5)
    before = outputs(model)
    aset.tensors("comfy")
    torch.testing.assert_close(outputs(model), before)
