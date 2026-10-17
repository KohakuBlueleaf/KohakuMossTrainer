"""T-LoRA masks: LyCORIS's per-timestep rule per sample, one mask group per token axis,
installed by begin_step and gone after end_step (sampling and exports see full rank)."""

from contextlib import nullcontext

import pytest
import torch
from _adtoy import perturb
from lycoris.modules.tlora import compute_timestep_mask, get_timestep_mask

from kohakumoss.adapters import (
    AdapterSpec,
    adapter_set,
    apply,
    network,
    tlora,
    token_mask,
)
from kohakumoss.model import random_backbone
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.negative import must_fail
from tests.model.conftest import HWS, TEXT_LENS, make_batch

pytestmark = pytest.mark.cpu
TOL = dict(rtol=2e-3, atol=2e-3)


@pytest.fixture(autouse=True)
def _no_leftover_masks():
    yield
    tlora.clear_masks()


def test_token_mask_matches_lycoris_rule_per_sample():
    sigma = torch.tensor([0.0, 0.31, 0.77, 1.0])
    seg = torch.tensor([0, 0, 1, 2, 2, 2, 3], dtype=torch.int32)
    mask = token_mask(sigma, seg, max_rank=16, min_rank=2, alpha=1.5)
    for i, s in enumerate(sigma.tolist()):
        want = compute_timestep_mask(int(round(s * 1000)), 1000, 16, 2, 1.5)
        rows = mask[seg == i]
        assert torch.equal(rows, want.expand_as(rows)), i

    def shared_mask():
        flat = token_mask(sigma[:1].expand(4), seg, 16, 2, 1.5)
        assert torch.equal(flat, mask)

    must_fail(shared_mask, "one mask for the whole batch")


def _tlora_backbone(preset: str = "all", **kw):
    model = random_backbone("tiny", seed=3).requires_grad_(False)
    spec = AdapterSpec(algo="tlora", preset=preset, dim=8, alpha=4.0, **kw)
    aset = apply(model, spec)
    perturb(aset, std=1.0)
    model.eval()
    return model, aset


def _packed(model, aset, sigma=None):
    img, text, s, seq, lat, txt = make_batch(model.cfg)
    sigma = s if sigma is None else sigma
    aset.begin_step(sigma, seq, 0)
    with torch.no_grad():
        y = model(img, text, sigma, seq)
    return y, sigma, lat, txt


def _per_sample(model, aset, sigma, lat, txt):
    outs = []
    for i, (x, t) in enumerate(zip(lat, txt)):
        seq = build_seqinfo(torch.tensor([HWS[i]]), torch.tensor([TEXT_LENS[i]]))
        img = x.flatten(2).transpose(1, 2)[0]
        aset.begin_step(sigma[i : i + 1], seq, 1)
        with torch.no_grad():
            outs.append(model(img, t[0], sigma[i : i + 1], seq))
        aset.end_step()
    return torch.cat(outs)


def test_every_token_axis_reads_its_own_mask():
    model, aset = _tlora_backbone("all")
    axes = {t.axis for t in aset.targets}
    assert axes == {"joint", "text", "image", "sample"}, axes
    packed, sigma, lat, txt = _packed(model, aset)
    aset.end_step()
    torch.testing.assert_close(packed, _per_sample(model, aset, sigma, lat, txt), **TOL)


def test_one_joint_mask_for_every_layer_is_caught(monkeypatch):
    monkeypatch.setattr(network, "axis_of", lambda path: "joint")

    def joint_only():
        model, aset = _tlora_backbone("all")
        try:
            packed, sigma, lat, txt = _packed(model, aset)
        except RuntimeError as e:
            raise AssertionError(e) from e
        aset.end_step()
        per = _per_sample(model, aset, sigma, lat, txt)
        torch.testing.assert_close(packed, per, **TOL)

    must_fail(joint_only, "text / image / modulation layers on the joint mask")


def test_mask_changes_the_output():
    model, aset = _tlora_backbone("attn-mlp")
    with torch.no_grad():
        img, text, sigma, seq, _, _ = make_batch(model.cfg)
        full = model(img, text, sigma, seq)
    masked, _, _, _ = _packed(model, aset, sigma)
    assert (full - masked).abs().max() > 1e-2
    aset.end_step()
    with torch.no_grad():
        torch.testing.assert_close(model(img, text, sigma, seq), full, **TOL)


def test_exports_ignore_and_keep_a_live_mask(tmp_path):
    model, aset = _tlora_backbone("attn-mlp")
    _packed(model, aset)
    live = get_timestep_mask(tlora.AXES["joint"])
    aset.export(str(tmp_path / "m.safetensors"), "merged", dtype=torch.float32)
    aset.export(str(tmp_path / "c.safetensors"), "comfy", dtype=torch.float32)
    assert get_timestep_mask(tlora.AXES["joint"]) is live
    aset.end_step()
    assert get_timestep_mask(tlora.AXES["joint"]) is None


def test_merged_export_under_a_mask_needs_the_evaluated_scope(monkeypatch, tmp_path):
    def unscoped():
        with monkeypatch.context() as patch:
            patch.setattr(adapter_set, "evaluated", lambda targets: nullcontext())
            model, aset = _tlora_backbone("attn-mlp")
            _packed(model, aset)
            try:
                aset.export(str(tmp_path / "m.safetensors"), "merged")
            except ValueError as e:
                raise AssertionError(e) from e
            finally:
                aset.end_step()

    must_fail(unscoped, "a merged export reading the per-token training mask")


def test_mask_settings_come_from_mask_keys():
    spec = AdapterSpec(algo="tlora", dim=8, algo_kwargs={"mask_alpha": 2.0})
    assert spec.mask_kwargs() == {"alpha": 2.0}
    assert "mask_alpha" not in spec.module_kwargs()
    assert not spec.validate()
