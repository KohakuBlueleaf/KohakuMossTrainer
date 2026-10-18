"""ComfyUI text-encoder files (`text_encoders/qwen3.5_9b_*.safetensors`): their key
layout, and the packaged Qwen3.5-9B config and tokenizer a ComfyUI file does not ship
(`ASSETS`, Apache-2.0, see its NOTICE). See docs/guides/weights.md#text-encoder."""

import gzip
import json
from pathlib import Path

from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

ASSETS = Path(__file__).resolve().parent / "assets" / "qwen3.5-9b"
COMFY_TEXT_PREFIX = "model.language_model."
COMFY_SKIPPED = ("model.visual.", "mtp.", "lm_head.")
MARK = COMFY_TEXT_PREFIX + "embed_tokens.weight"
LOAD_ONLY = ("tokenizer_class", "backend", "is_local", "local_files_only")


def packaged_config() -> PretrainedConfig:
    """The transformers config of the packaged Qwen3.5-9B `config.json` (the text model's
    is its `text_config`)."""
    return AutoConfig.from_pretrained(str(ASSETS))


def packaged_tokenizer():
    """The Qwen3.5 tokenizer built in memory from the packaged `tokenizer.json.gz`,
    `tokenizer_config.json` and `chat_template.jinja`: the tokenizer
    `AutoTokenizer.from_pretrained` gives for MOSS-Image's `tokenizer/` folder."""
    cfg = json.loads((ASSETS / "tokenizer_config.json").read_text(encoding="utf-8"))
    raw = gzip.decompress((ASSETS / "tokenizer.json.gz").read_bytes())
    template = (ASSETS / "chat_template.jinja").read_text(encoding="utf-8")
    cls = tokenizer_class_from_name(cfg["tokenizer_class"])
    kwargs = {k: v for k, v in cfg.items() if k not in LOAD_ONLY}
    return cls(
        tokenizer_object=Tokenizer.from_str(raw.decode("utf-8")),
        chat_template=template,
        **kwargs,
    )


def comfy_text_key(file_key: str) -> str | None:
    """The text model's own key of a ComfyUI text-encoder file key, or None for the
    parts the text encoder does not use (`COMFY_SKIPPED`: vision tower, MTP head,
    LM head); KeyError for a key outside the layout."""
    if file_key.startswith(COMFY_SKIPPED):
        return None
    if file_key.startswith(COMFY_TEXT_PREFIX):
        return file_key[len(COMFY_TEXT_PREFIX) :]
    raise KeyError(f"{file_key!r} is not a key of a ComfyUI Qwen3.5 text-encoder file")


def is_comfy_text_file(path: str | Path) -> bool:
    """Whether `path` is a ComfyUI Qwen3.5 text-encoder file (its header holds
    `model.language_model.embed_tokens.weight`); no tensor is read."""
    path = Path(path)
    if not path.is_file() or path.suffix != ".safetensors":
        return False
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return MARK in f.keys()
