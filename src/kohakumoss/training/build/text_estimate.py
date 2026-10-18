"""The online text encoder's part of the `auto` memory plan, from its config alone (a
weightless meta-device twin under the TextEncoderPlan; no weight is read)."""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from kohakumoss.encoders.text.plan import DTYPES, resolve_dtype
from kohakumoss.offload.encoders.text_memory import TextSizes, text_sizes
from kohakumoss.quant.text import quantize_text_linears

DEFAULT_PASS_TOKENS = 16384


def text_model_dir(path: str | Path) -> Path:
    """The text model directory of a pipeline root (text_encoder/) or a model dir."""
    path = Path(path)
    return path / "text_encoder" if (path / "text_encoder").is_dir() else path


def meta_text_model(config, dtype: torch.dtype, plan) -> torch.nn.Module:
    """The text model of `config` on the meta device in `dtype`, its linears quantized
    as `plan.precision` stores them."""
    with torch.device("meta"):
        model = AutoModel.from_config(config).to(dtype)
    text = getattr(model, "language_model", model)
    if plan.precision != "compute":
        quantize_text_linears(
            text, plan.precision, plan.quant_group, dtype, recipe=plan.fp8_recipe
        )
    return text


def planned_text_sizes(path: str | Path, plan, device: torch.device) -> TextSizes:
    """TextSizes of the text encoder at `path` placed by `plan` on `device` (the plan's
    resolved dtype and precision, its stream ring of `stream_resident` slots)."""
    plan = resolve_dtype(plan, device)
    config = AutoConfig.from_pretrained(str(text_model_dir(path)))
    text = meta_text_model(config, DTYPES[plan.dtype], plan)
    return text_sizes(text, plan.stream_resident)


def pass_tokens(plan) -> tuple[int, int]:
    """(tokens per pass, tokens per call) the estimate assumes: the plan's max_tokens,
    else DEFAULT_PASS_TOKENS."""
    tokens = plan.max_tokens or DEFAULT_PASS_TOKENS
    return tokens, tokens
