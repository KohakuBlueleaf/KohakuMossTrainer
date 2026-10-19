"""The online text encoder's part of the `auto` memory plan, from its config alone (a
weightless meta-device twin under the TextEncoderPlan; no weight is read)."""

from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from kohakumoss.encoders.text.comfy_files import is_comfy_text_file, packaged_config
from kohakumoss.encoders.text.encoder import resolve_text_plan, text_source_of
from kohakumoss.encoders.text.plan import DTYPES
from kohakumoss.offload.encoders.text_memory import TextSizes, text_sizes
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.text import quantize_text_linears

DEFAULT_PASS_TOKENS = 16384
STORED_AS_INT8 = ("int8", "w4a8")


def text_model_dir(path: str | Path) -> Path:
    """The text model directory of a pipeline root (text_encoder/) or a model dir."""
    path = Path(path)
    return path / "text_encoder" if (path / "text_encoder").is_dir() else path


def _int8_embedding(text: nn.Module, dtype: torch.dtype) -> None:
    emb = text.embed_tokens
    with torch.device("meta"):
        q = torch.empty(emb.num_embeddings, emb.embedding_dim, dtype=torch.int8)
        scale = torch.empty(emb.num_embeddings)
    text.embed_tokens = QuantEmbedding(q, scale, 0, dtype)


def meta_text_model(config, dtype: torch.dtype, plan) -> torch.nn.Module:
    """The text model of `config` on the meta device in `dtype`, its linears quantized
    as `plan.precision` stores them (a stored W4A8 file at its int8 decode, the
    larger form; a stored int8 / W4A8 file's int8 embedding table)."""
    with torch.device("meta"):
        model = AutoModel.from_config(config).to(dtype)
    text = getattr(model, "language_model", model)
    precision = "int8" if plan.precision == "w4a8" else plan.precision
    if precision != "compute":
        quantize_text_linears(
            text, precision, plan.quant_group, dtype, recipe=plan.fp8_recipe
        )
    return text


def planned_text_sizes(path: str | Path, plan, device: torch.device) -> TextSizes:
    """TextSizes of the text encoder at `path` (a model dir, a pipeline root, a ComfyUI
    file or folder) placed by `plan` on `device` (the plan's resolved dtype and
    precision, its stream ring of `stream_resident` slots)."""
    path = text_source_of(path)
    plan, stored, _ = resolve_text_plan(path, plan, device)
    dtype = DTYPES[plan.dtype]
    if is_comfy_text_file(path):
        text = meta_text_model(packaged_config().text_config, dtype, plan)
        if plan.precision == stored and stored in STORED_AS_INT8:
            _int8_embedding(text, dtype)
    else:
        config = AutoConfig.from_pretrained(str(text_model_dir(path)))
        text = meta_text_model(config, dtype, plan)
    return text_sizes(text, plan.stream_resident)


def pass_tokens(plan) -> tuple[int, int]:
    """(tokens per pass, tokens per call) the estimate assumes: the plan's max_tokens,
    else DEFAULT_PASS_TOKENS."""
    tokens = plan.max_tokens or DEFAULT_PASS_TOKENS
    return tokens, tokens
