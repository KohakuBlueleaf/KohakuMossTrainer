"""Preset expressions, toml presets and LyCORIS map semantics select the right layers."""

import pytest
from _adtoy import toy

from kohakumoss.adapters import AdapterSpec, apply, presets
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _paths(preset, **kw):
    model = toy()
    aset = apply(model, AdapterSpec(preset=preset, dim=2, alpha=1.0, **kw))
    return {t.path for t in aset.targets}, {n.path for n in aset.norms}, aset


def test_attn_preset_hits_joint_qkvo_only():
    paths, _, _ = _paths("attn")
    expected = {f"layers.{i}.proj.{p}" for i in range(2) for p in "qkvo"}
    assert paths == expected


def test_attn_mlp_and_modifiers():
    paths, _, _ = _paths("attn-mlp+text-refiner+modulation")
    assert "layers.1.proj.ffn_down" in paths
    assert "text_embedder.blocks.0.qkv" in paths
    assert "text_embedder.proj" in paths
    assert "shared_block_modulation" in paths and "time_embedding.2" in paths
    assert "video_embedder" not in paths


def test_all_preset_covers_every_linear():
    model = toy()
    linears = {n for n, m in model.named_modules() if type(m).__name__ == "MossLinear"}
    paths, _, _ = _paths("all")
    assert paths == linears


def test_train_norms_follow_the_preset():
    _, norms, aset = _paths("attn", train_norms=True)
    assert norms == {
        f"layers.{i}.{n}"
        for i in range(2)
        for n in ("norm1", "norm2", "q_norm", "k_norm")
    }
    assert all(p.requires_grad for p in aset.trainable_params())


def test_lora_names_follow_the_prefix():
    _, _, aset = _paths("attn")
    assert aset.targets[0].lora_name.startswith("lora_unet_layers_")
    assert {t.lora_name for t in aset.targets} == {
        "lora_unet_" + t.path.replace(".", "_") for t in aset.targets
    }


def test_dict_preset_with_name_algo_map_and_exclude():
    preset = {
        "target_name": [r"layers\.\d+\.proj\.(q|v)$"],
        "name_algo_map": {r"layers\.1\.proj\.v$": {"algo": "loha", "dim": 2}},
        "exclude_name": [r"layers\.0\.proj\.v$"],
    }
    paths, _, aset = _paths(preset)
    assert paths == {"layers.0.proj.q", "layers.1.proj.q", "layers.1.proj.v"}
    kinds = {t.path: t.kind for t in aset.targets}
    assert kinds["layers.1.proj.v"] == "loha" and kinds["layers.0.proj.q"] == "locon"


def test_toml_preset(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('target_name = ["layers\\\\.0\\\\.proj\\\\.o$"]\n')
    paths, _, _ = _paths(str(path))
    assert paths == {"layers.0.proj.o"}


def _leak_round():
    first, _, _ = _paths(
        {
            "target_name": [r"layers\.0\.proj\.(o|q)$"],
            "exclude_name": [r"layers\.0\.proj\.q$"],
        }
    )
    second, _, _ = _paths("attn")
    assert first == {"layers.0.proj.o"}
    assert "layers.0.proj.q" in second and len(second) == 8


def test_presets_do_not_leak_between_calls(monkeypatch):
    _leak_round()
    with monkeypatch.context() as patch:
        patch.setattr(presets, "PRESET_KEYS_DEFAULT", {})
        must_fail(_leak_round, "a preset that leaves exclude_name to the previous call")


def test_unknown_preset_is_an_error():
    with pytest.raises(KeyError):
        _paths("attn+nonsense")


def test_double_apply_is_refused():
    model = toy()
    apply(model, AdapterSpec(preset="attn", dim=2))
    with pytest.raises(RuntimeError):
        apply(model, AdapterSpec(preset="attn", dim=2))
