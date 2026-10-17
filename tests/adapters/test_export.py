"""Exports: ComfyUI evaluates them to the trained dW; files are deterministic witnesses."""

import hashlib
import json

import pytest
import torch
from _adcases import CASES, build, make_spec
from _adref import comfy_weight, lycoris_delta
from _adtoy import perturb
from safetensors import safe_open
from safetensors.torch import load_file

from kohakumoss.adapters import apply
from kohakumoss.model import random_backbone
from kohakumoss.model.io.keymap import fuse_qkv as fuse_layers
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
TOL = dict(rtol=1e-4, atol=1e-6)
NATIVE = {"locon", "loha", "lokr", "glora", "oft", "boft", "ia3", "full"}


def comfy_key_map(state: dict) -> dict:
    """ComfyUI's MOSS key map: lora_unet_ / diffusion_model. stems, fused q/k/v slices."""
    out = {}
    for k, v in state.items():
        if not k.endswith(".weight"):
            continue
        stem = k[: -len(".weight")]
        out["lora_unet_" + stem.replace(".", "_")] = (k, None)
        out["diffusion_model." + stem] = (k, None)
        if stem.endswith(".proj.qkv"):
            third = v.shape[0] // 3
            for i, name in enumerate("qkv"):
                part = stem[: -len("qkv")] + name
                out["lora_unet_" + part.replace(".", "_")] = (k, (i * third, third))
                out["diffusion_model." + part] = (k, (i * third, third))
    return out


def comfy_patch(state: dict, lora: dict) -> dict:
    """Apply a LoRA file the way ComfyUI does (strength 1), returning a new state."""
    key_map = comfy_key_map(state)
    stems: dict[str, dict] = {}
    for key, value in lora.items():
        for stem in key_map:
            tail = key[len(stem) + 1 :]
            if key.startswith(stem + ".") and tail.split(".")[0] in _TAILS:
                stems.setdefault(stem, {})[tail] = value
    out = {k: v.double().clone() for k, v in state.items()}
    for stem, parts in stems.items():
        target, sl = key_map[stem]
        w = out[target] if sl is None else out[target].narrow(0, *sl)
        patched = comfy_weight(parts, w)
        if sl is None:
            out[target] = patched
        else:
            out[target].narrow(0, *sl).copy_(patched)
    return out


_TAILS = {
    "lora_up", "lora_down", "alpha", "dora_scale", "hada_w1_a", "hada_w1_b",
    "hada_w2_a", "hada_w2_b", "lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2",
    "lokr_w2_a", "lokr_w2_b", "oft_blocks", "rescale", "a1", "a2", "b1", "b2",
    "diff", "diff_b", "w_norm", "b_norm",
}  # fmt: skip


def fuse_qkv(state: dict, depth: int = 2) -> dict:
    return fuse_layers(state, depth)


def backbone_set(name: str, **override):
    """A tiny MossBackbone with every adapter tensor randomised."""
    model = random_backbone("tiny", seed=3)
    model.requires_grad_(False)
    aset = apply(model, make_spec(name, **override))
    perturb(aset)
    model.eval()
    return model, aset


def _base_state(aset):
    return {f"{t.path}.weight": t.base_weight(torch.float64) for t in aset.targets}


def _expected(aset):
    return {
        f"{t.path}.weight": t.base_weight(torch.float64) + t.delta()[0].double()
        for t in aset.targets
    }


@pytest.mark.parametrize("style", ["lora_unet", "diffusion_model"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_comfy_export_evaluates_to_trained_delta(name, style):
    _, aset = build(name)
    lora = aset.tensors("comfy", style)
    patched = comfy_patch(_base_state(aset), lora)
    for key, want in _expected(aset).items():
        torch.testing.assert_close(patched[key], want, **TOL, msg=f"{name} {key}")


def test_comfy_export_onto_fused_qkv_file():
    _, aset = build("lora")
    split = _base_state(aset)
    fused = fuse_qkv(split)
    assert "layers.0.proj.qkv.weight" in fused and "layers.0.proj.q.weight" not in fused
    patched = comfy_patch(fused, aset.tensors("comfy"))
    want = fuse_qkv(_expected(aset))
    for key in want:
        torch.testing.assert_close(patched[key], want[key].double(), **TOL)


@pytest.mark.parametrize("name", ["oft-constraint", "dora-out", "ia3"])
def test_comfy_conversions_are_needed(name):
    _, aset = build(name)
    native_as_comfy = {
        f"diffusion_model.{t.path}.{k}": v
        for t in aset.targets
        for k, v in t.lyco.state_dict().items()
    }

    def unconverted():
        patched = comfy_patch(_base_state(aset), native_as_comfy)
        for key, want in _expected(aset).items():
            torch.testing.assert_close(patched[key], want, **TOL)

    if name == "ia3":
        with pytest.raises(Exception):
            unconverted()
        return
    must_fail(unconverted, f"{name} exported without the ComfyUI conversion")


@pytest.mark.parametrize(
    "name",
    sorted(
        n
        for n in CASES
        if CASES[n]["algo"]
        in ("lora", "loha", "lokr", "glora", "oft", "boft", "ia3", "full")
    ),
)
def test_native_file_matches_independent_formula(name, tmp_path):
    _, aset = build(name)
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "lycoris", dtype=None)
    native = load_file(str(path))
    for t in aset.targets:
        if t.kind not in NATIVE:
            continue
        parts = {
            k[len(t.lora_name) + 1 :]: v
            for k, v in native.items()
            if k.startswith(t.lora_name + ".")
        }
        w = t.base_weight(torch.float64)
        ref = lycoris_delta(
            t.kind if t.kind != "locon" else "locon", parts, w, w.shape[0]
        )
        torch.testing.assert_close(t.delta()[0].double(), ref, **TOL)


def _sha(path) -> str:
    """Hash of every tensor's name, dtype, shape and bytes (the header order is not fixed)."""
    h = hashlib.sha256()
    for key, value in sorted(load_file(str(path)).items()):
        h.update(f"{key}|{value.dtype}|{tuple(value.shape)}".encode())
        h.update(value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def test_export_bytes_are_a_witness(tmp_path):
    _, aset = build("lora")
    a, b, c = (tmp_path / f"{i}.safetensors" for i in "abc")
    aset.export(str(a), "comfy")
    aset.export(str(b), "comfy")
    assert _sha(a) == _sha(b)
    t = aset.targets[3]
    with torch.no_grad():
        t.lyco.lora_up.weight[0, 0] += 0.5
    aset.export(str(c), "comfy")
    assert _sha(a) != _sha(c)
    got = load_file(str(c))[f"lora_unet_{t.path.replace('.', '_')}.lora_up.weight"]
    torch.testing.assert_close(got, t.lyco.lora_up.weight.detach().to(got.dtype))


def test_metadata_records_format_and_args(tmp_path):
    _, aset = build("lokr")
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy", metadata={"note": "x"})
    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
    assert meta["kohakumoss_format"] == "comfy" and meta["note"] == "x"
    assert json.loads(meta["ss_network_args"])["algo"] == "lokr"


@pytest.mark.parametrize("fused", [False, True])
def test_merged_checkpoint(fused, tmp_path):
    model, aset = backbone_set("lora-norms")
    path = tmp_path / "m.safetensors"
    aset.export(str(path), "merged", dtype=torch.float32, fused_qkv=fused)
    state = load_file(str(path))
    assert not any(".lyco." in k for k in state)
    want = _expected(aset)
    if fused:
        want = fuse_qkv(want, model.cfg.depth)
    for key, value in want.items():
        torch.testing.assert_close(state[key].double(), value.double(), **TOL)
    norm = aset.norms[0]
    torch.testing.assert_close(
        state[f"{norm.path}.weight"], norm.module.weight.detach()
    )
    assert "video_embedder.weight" in state and "final_linear_video.weight" in state
    assert ("layers.0.proj.qkv.weight" in state) == fused
    assert ("layers.0.proj.q.weight" in state) != fused


def test_diffusers_shards(tmp_path):
    model, aset = backbone_set("lora")
    out = tmp_path / "transformer"
    aset.export(str(out), "diffusers", dtype=torch.float32)
    index = json.loads(
        (out / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    merged = {}
    for name in set(index["weight_map"].values()):
        merged.update(load_file(str(out / name)))
    for key, value in _expected(aset).items():
        torch.testing.assert_close(merged[key].double(), value, **TOL)
    config = json.loads((out / "config.json").read_text())
    assert config["dim"] == model.cfg.dim and config["num_layers"] == model.cfg.depth


def test_row_scale_uniform_and_per_row():
    _, aset = build("lokr")
    t = next(t for t in aset.targets if t.path == "layers.0.proj.q")
    base = _base_state(aset)
    uniform = aset.tensors("comfy", row_scales={t.path: 2.0})
    rows = torch.linspace(0.5, 1.5, t.lyco.shape[0])
    per_row = aset.tensors("comfy", row_scales={t.path: 1.0 / rows})
    key = f"{t.path}.weight"
    want = t.delta()[0].double()
    p1 = comfy_patch(base, uniform)[key] - base[key]
    p2 = comfy_patch(base, per_row)[key] - base[key]
    torch.testing.assert_close(p1, want * 0.5, **TOL)
    torch.testing.assert_close(p2, want * rows.double()[:, None], **TOL)
