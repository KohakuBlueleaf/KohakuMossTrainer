"""Dotted text-encoder factory for the offline text cache (scripts/data/cache_text.py)."""

from pathlib import Path

import torch

from kohakumoss.encoders.specs import text_spec
from kohakumoss.encoders.text.encoder import QwenTextEncoder, resolve_text_plan
from kohakumoss.encoders.text.plan import DTYPES, text_plan
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_kernels.dtypes import NAME_OF

PRECOMPUTE_DEFAULTS = {"memory": "auto", "batch": 32, "max_tokens": 16384}


def factory_plan(dtype=None, plan: dict | None = None) -> TextEncoderPlan:
    """The TextEncoderPlan `text_encoder` places the encoder by: PRECOMPUTE_DEFAULTS,
    then `plan`, then `dtype` when given."""
    spec = {**PRECOMPUTE_DEFAULTS, **(plan or {})}
    if dtype is not None:
        spec["dtype"] = NAME_OF[dtype] if isinstance(dtype, torch.dtype) else dtype
    return text_plan(spec)


def text_encoder_spec(
    path: str | Path = "",
    device: torch.device | str = "cuda",
    dtype: torch.dtype | str | None = None,
    plan: dict | None = None,
    max_length: int = 4096,
) -> str:
    """The `encoder_spec` `text_encoder(...)` gives for the same arguments, without
    loading the weights (their headers only)."""
    spec = factory_plan(dtype, plan)
    on = torch.device("cpu") if spec.memory == "cpu" else torch.device(device)
    spec, _, layout = resolve_text_plan(path, spec, on)
    return text_spec(spec, DTYPES[spec.dtype], layout)


def text_encoder(
    path: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | str | None = None,
    plan: dict | None = None,
    max_length: int = 4096,
) -> QwenTextEncoder:
    """An EncodeText callable (with `encoder_id`) over the real Qwen3.5 weights at
    `path` (pipeline root or model dir); `dtype` ("bf16" | "fp16" or a torch dtype,
    default the plan's) and `plan` (TextEncoderPlan fields) place it; an unset memory
    is "auto" (the fastest placement that fits the device). Its outputs land on the
    host; the plan is checked for the device before the weights load."""
    if not str(path):
        raise ValueError(
            "text_encoder: path is empty; set TEXT_ENCODER_KWARGS['path'] to the text "
            "encoder dir or a pipeline root"
        )
    if str(device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the text-encoder factory defaults to CUDA; pass device=")
    return QwenTextEncoder.from_pretrained(
        path, device, factory_plan(dtype, plan), max_length, output_device="cpu"
    )
