"""ComfyUI text-encoder files (model-free): the key map onto the text model's own keys,
the header check on constructed files, and the packaged Qwen3.5-9B config against the
values of MOSS-Image's `text_encoder/config.json`."""

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.encoders.text.comfy_files import (
    ASSETS,
    COMFY_SKIPPED,
    comfy_text_key,
    is_comfy_text_file,
    packaged_config,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]
TEXT = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 32,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_value_heads": 32,
    "linear_num_key_heads": 16,
    "layer_types": LAYER_TYPES * 8,
}
CASES = {
    "model.language_model.embed_tokens.weight": "embed_tokens.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight": "layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight_scale": "layers.0.linear_attn.out_proj.weight_scale",
    "model.language_model.layers.0.mlp.up_proj.comfy_quant": "layers.0.mlp.up_proj.comfy_quant",
    "model.language_model.norm.weight": "norm.weight",
    "model.visual.blocks.0.attn.qkv.weight": None,
    "mtp.fc.weight": None,
    "lm_head.weight": None,
}


def check_keys(mapper) -> None:
    for file_key, want in CASES.items():
        assert mapper(file_key) == want, (file_key, mapper(file_key), want)
    with pytest.raises(KeyError):
        mapper("language_model.embed_tokens.weight")


def _unstripped(key: str) -> str | None:
    return None if key.startswith(COMFY_SKIPPED) else key


def test_comfy_text_keys():
    check_keys(comfy_text_key)
    must_fail(lambda: check_keys(_unstripped), "the prefix kept")


def check_config(text) -> None:
    for name, value in TEXT.items():
        assert getattr(text, name) == value, (name, getattr(text, name), value)


def test_packaged_config():
    cfg = packaged_config()
    assert cfg.model_type == "qwen3_5"
    check_config(cfg.text_config)
    broken = packaged_config().text_config
    broken.layer_types = list(reversed(broken.layer_types))
    must_fail(lambda: check_config(broken), "layer types reversed")
    for name in ("LICENSE", "NOTICE", "tokenizer.json.gz", "chat_template.jinja"):
        assert (ASSETS / name).is_file(), name


def test_comfy_text_file_by_header(tmp_path):
    comfy = tmp_path / "comfy.safetensors"
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.zeros(1),
            "lm_head.weight": torch.zeros(1),
        },
        str(comfy),
    )
    hf = tmp_path / "hf.safetensors"
    save_file({"language_model.embed_tokens.weight": torch.zeros(1)}, str(hf))
    assert is_comfy_text_file(comfy)
    assert not is_comfy_text_file(hf)
    assert not is_comfy_text_file(tmp_path)
    must_fail(lambda: _assert_comfy(hf), "an HF-layout file")


def _assert_comfy(path) -> None:
    assert is_comfy_text_file(path)
