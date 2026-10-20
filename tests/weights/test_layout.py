"""Weights layouts (model-free, empty files): a ComfyUI models folder is recognised
and each component resolves to its default file (int8 ConvRot first, the bf16 DiT first
for a full finetune, the next default when one is missing), a file name or a path; the
HF layout keeps its folders."""

import pytest

from kohakumoss.weights import (
    DIT,
    TEXT,
    VAE,
    comfy_defaults,
    comfy_file,
    component_path,
    hf_root,
    is_comfy_folder,
    layout_of,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
FILES = {
    "diffusion_models": (
        "moss_image_base_bf16",
        "moss_image_base_int8_convrot",
        "moss_image_sft_bf16",
    ),
    "text_encoders": ("qwen3.5_9b_bf16", "qwen3.5_9b_w4a8_gptq"),
    "vae": ("flux2-vae", "flux2-vae_small"),
}


def _comfy(root):
    for sub, names in FILES.items():
        (root / sub).mkdir(parents=True)
        for name in names:
            (root / sub / f"{name}.safetensors").write_bytes(b"")
    return root


def check_defaults(root) -> None:
    assert component_path(root, DIT).name == "moss_image_base_int8_convrot.safetensors"
    assert (
        component_path(root, DIT, full=True).name == "moss_image_base_bf16.safetensors"
    )
    assert component_path(root, DIT, variant="sft").name == "moss_image_sft_bf16.safetensors"  # fmt: skip
    assert component_path(root, TEXT).name == "qwen3.5_9b_bf16.safetensors"
    assert component_path(root, VAE).name == "flux2-vae_small.safetensors"


def test_comfy_folder_defaults_and_choices(tmp_path):
    root = _comfy(tmp_path / "models")
    assert is_comfy_folder(root) and layout_of(root) == "comfy"
    check_defaults(root)
    assert comfy_file(root, TEXT, "qwen3.5_9b_w4a8_gptq").parent.name == "text_encoders"
    assert (
        comfy_file(root, VAE, "flux2-vae.safetensors").name == "flux2-vae.safetensors"
    )
    elsewhere = tmp_path / "mine.safetensors"
    elsewhere.write_bytes(b"")
    assert component_path(root, DIT, str(elsewhere)) == elsewhere
    with pytest.raises(FileNotFoundError, match="moss_image_xl"):
        component_path(root, DIT, "moss_image_xl")
    assert comfy_defaults(TEXT)[0] == "qwen3.5_9b_int8_convrot.safetensors"
    (root / "vae" / "flux2-vae_small.safetensors").unlink()
    must_fail(lambda: check_defaults(root), "the small VAE removed")
    assert component_path(root, VAE).name == "flux2-vae.safetensors"


def test_hf_layout_is_unchanged(tmp_path):
    pipe = tmp_path / "MOSS-Image-Base"
    for sub in ("transformer", "text_encoder", "tokenizer", "vae"):
        (pipe / sub).mkdir(parents=True)
    (pipe / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    assert not is_comfy_folder(pipe) and layout_of(pipe) == "hf"
    assert component_path(pipe, DIT) == pipe
    assert component_path(pipe, TEXT) == pipe / "text_encoder"
    assert component_path(pipe, VAE, str(pipe / "vae")) == pipe / "vae"
    single = tmp_path / "dit.safetensors"
    single.write_bytes(b"")
    assert layout_of(single) == "file" and component_path(single, DIT) == single
    must_fail(lambda: _assert_comfy(pipe), "an HF folder")


def _assert_comfy(path) -> None:
    assert is_comfy_folder(path)


def _weights_root(root):
    """An HF weights root: base/ with every component, sft/ with its own DiT only."""
    for sub in ("base/transformer", "base/text_encoder", "base/vae", "sft/transformer"):
        (root / sub).mkdir(parents=True)
        (root / sub / "model.safetensors").write_bytes(b"")
    return root


def assert_root_resolved(path_of, root) -> None:
    assert path_of(root, DIT, variant="sft") == root / "sft"
    assert path_of(root, TEXT, variant="sft") == root / "base" / "text_encoder"
    assert path_of(root, VAE) == root / "base" / "vae"
    assert path_of(root, DIT) == root / "base"
    assert hf_root(root, TEXT) == root / "base"
    assert hf_root(root / "base" / "vae", VAE) == root / "base" / "vae"


def _flat(weights, component, choice="", variant="base", full=False):
    return weights if component == DIT else weights / component


def test_hf_weights_root_resolves_its_variant(tmp_path):
    root = _weights_root(tmp_path / "MOSS-Image")
    assert_root_resolved(component_path, root)
    must_fail(lambda: assert_root_resolved(_flat, root), "the root taken as a pipeline")
