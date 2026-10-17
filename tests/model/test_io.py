"""Checkpoint I/O against the official-layout oracle: HF keys, ComfyUI fused qkv, savers."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from kohakumoss.model import build_backbone, load_backbone, save_diffusers, save_single
from kohakumoss.model.io import load as load_module
from kohakumoss.model.io import source as source_module
from kohakumoss.model.io.load import infer_config, load_state_into, read_state
from kohakumoss.model.io.save import export_state
from kohakumoss.model.numerics import fold
from tests._util.negative import must_fail
from tests.model._official import OfficialTransformer
from tests.model.conftest import make_batch, oracle_outputs

pytestmark = pytest.mark.cpu


def _oracle(cfg, seed=5):
    torch.manual_seed(seed)
    oracle = OfficialTransformer(cfg)
    with torch.no_grad():
        for p in oracle.parameters():
            p.normal_(0, 0.05)
    return oracle


def _official_state(oracle):
    sd = dict(oracle.state_dict())
    sd["rope.bands"] = oracle.rope_bands.clone()
    return sd


def _check_against(model, oracle, cfg):
    img, text, sigma, seq, lat, txt = make_batch(cfg)
    out = model(img, text, sigma, seq)
    ref = oracle_outputs(oracle, lat, txt, sigma)
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 2e-5, rel


def _load(path, cfg):
    model, report = load_backbone(path, device="cpu", dtype=torch.float32, keep_fp32=())
    assert report.ok and _shape(model.cfg) == _shape(cfg)
    return model, report


def _shape(cfg):
    return cfg.override(negative_skip_layers=())


def test_load_streams_tensors(cfg, tmp_path, monkeypatch):
    """load_backbone never reads a whole file; a fused qkv tensor is read once."""
    oracle = _oracle(cfg)
    sd = dict(_official_state(oracle))
    for i in range(cfg.depth):
        base = f"layers.{i}.proj"
        sd[f"{base}.qkv.weight"] = torch.cat(
            [sd.pop(f"{base}.{n}.weight") for n in "qkv"], 0
        )
    save_file(sd, str(tmp_path / "f.safetensors"))

    def whole_file(*args, **kwargs):
        raise AssertionError("whole-file read")

    monkeypatch.setattr(load_module, "load_file", whole_file)
    reads = []
    real_get = source_module.CheckpointSource.get

    def counted(self, key):
        reads.append(key)
        return real_get(self, key)

    monkeypatch.setattr(source_module.CheckpointSource, "get", counted)
    model, _ = _load(tmp_path / "f.safetensors", cfg)
    _check_against(model, oracle, cfg)
    assert sorted(reads) == sorted(k for k in sd if k != "rope.bands")
    must_fail(lambda: read_state(tmp_path / "f.safetensors"), "whole-file read")


class _Stored:
    def __init__(self, dense):
        self.dense = dense
        self.shape = dense.shape

    def weight(self, dtype):
        return self.dense.to(dtype)

    def materialize(self):
        return self.dense


def test_export_state_overrides_and_stored_bases(tiny):
    fold(tiny, 2.0**-2, "gate")
    lin = tiny.layers[1].proj["q"]
    dense = torch.randn_like(lin.weight)
    lin.set_base(_Stored(dense))
    merged = torch.randn_like(tiny.video_embedder.weight)
    out = export_state(tiny, dtype=None, weights={"video_embedder.weight": merged})
    assert torch.equal(out["layers.1.proj.q.weight"], dense)
    assert torch.equal(out["video_embedder.weight"], merged * 4)
    plain = export_state(tiny, dtype=None)
    must_fail(
        lambda: _check(torch.equal(plain["video_embedder.weight"], merged * 4)),
        "weights ignored",
    )


def _check(cond):
    assert cond


def test_hf_single_file(cfg, tmp_path):
    oracle = _oracle(cfg)
    save_file(_official_state(oracle), str(tmp_path / "w.safetensors"))
    model, report = _load(tmp_path / "w.safetensors", cfg)
    assert report.prefix == "" and report.derived == []
    _check_against(model, oracle, cfg)


def test_comfy_fused_with_prefix(cfg, tmp_path):
    oracle = _oracle(cfg)
    sd = {k: v for k, v in oracle.state_dict().items()}
    for i in range(cfg.depth):
        base = f"layers.{i}.proj"
        sd[f"{base}.qkv.weight"] = torch.cat(
            [sd.pop(f"{base}.{n}.weight") for n in "qkv"], 0
        )
    sd = {f"model.diffusion_model.{k}": v for k, v in sd.items()}
    save_file(sd, str(tmp_path / "c.safetensors"))
    model, report = _load(tmp_path / "c.safetensors", cfg)
    assert report.prefix == "model.diffusion_model." and report.derived == [
        "rope.bands"
    ]
    _check_against(model, oracle, cfg)


def test_strict_key_accounting(cfg):
    oracle = _oracle(cfg)
    model = build_backbone(cfg, device="meta", dtype=torch.float32, keep_fp32=())
    missing = _official_state(oracle)
    missing.pop("layers.1.proj.o.weight")
    must_fail(
        lambda: _expect_raise(lambda: load_state_into(model, missing)), "missing key"
    )
    with pytest.raises(KeyError):
        load_state_into(model, missing)
    extra = _official_state(oracle)
    extra["layers.0.proj.bogus"] = torch.zeros(1)
    with pytest.raises(KeyError):
        load_state_into(model, extra)


def _expect_raise(fn):
    try:
        fn()
    except KeyError:
        raise AssertionError("raised")


def test_infer_config(cfg):
    hf = {"rope_base": cfg.rope_base}
    assert _shape(infer_config(_official_state(_oracle(cfg)), hf)) == _shape(cfg)


def test_save_single_split_and_fused(cfg, tmp_path):
    oracle = _oracle(cfg)
    save_file(_official_state(oracle), str(tmp_path / "w.safetensors"))
    model, _ = _load(tmp_path / "w.safetensors", cfg)
    fold(model, 2.0**-3)
    save_single(model, tmp_path / "split.safetensors", dtype=None)
    split = load_file(str(tmp_path / "split.safetensors"))
    for k, v in _official_state(oracle).items():
        assert torch.equal(split[k], v), k
    fused = export_state(model, dtype=None, fused_qkv=True)
    q = oracle.state_dict()["layers.2.proj.q.weight"]
    k = oracle.state_dict()["layers.2.proj.k.weight"]
    v = oracle.state_dict()["layers.2.proj.v.weight"]
    assert torch.equal(fused["layers.2.proj.qkv.weight"], torch.cat((q, k, v)))
    assert "layers.2.proj.q.weight" not in fused


def test_save_diffusers(cfg, tmp_path):
    oracle = _oracle(cfg)
    save_file(_official_state(oracle), str(tmp_path / "w.safetensors"))
    model, _ = _load(tmp_path / "w.safetensors", cfg)
    save_diffusers(
        model, tmp_path / "transformer", dtype=torch.float32, shard_bytes=1 << 16
    )
    index = json.loads(
        (
            tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        ).read_text()
    )
    assert len(set(index["weight_map"].values())) > 1
    back = read_state(tmp_path)
    for k, v in _official_state(oracle).items():
        assert torch.equal(back[k], v), k
    config = json.loads((tmp_path / "transformer" / "config.json").read_text())
    assert config["num_layers"] == cfg.depth and config["dim"] == cfg.dim
