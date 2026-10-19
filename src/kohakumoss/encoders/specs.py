"""Encoder compute specs: the part of a cache key that names how an encoder computed
(VAE dtype; text-encoder dtype, weight precision and quantization layout), and the
encode callable that carries its spec. See docs/concepts/data.md#cache-keys."""

import inspect

import torch

from kohakumoss.quant.compute import COMPUTE, recipe_of
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_common.imports import import_object
from kohakumoss_kernels.dtypes import NAME_OF

WEIGHT_ONLY = ("int8", "fp8")
TEXT_SPEC_FIELDS = ("dtype", "precision", "quant_group", "fp8_recipe")


def dtype_name(dtype: torch.dtype) -> str:
    """The spec name of `dtype`: NAME_OF's short name, else torch's own name (the
    precision reference's dtype)."""
    return NAME_OF.get(dtype) or str(dtype).removeprefix("torch.")


def vae_spec(dtype: torch.dtype) -> str:
    """The VAE's compute dtype: `fp32` | `bf16` | `fp16`."""
    return dtype_name(dtype)


def text_spec(plan: TextEncoderPlan, dtype: torch.dtype, layout: str = "") -> str:
    """`<dtype>/<precision>`, plus `g<quant_group>` for a weight-only int8 / fp8 base or
    the kernel recipe for fp8 compute, plus `src:<layout>` when the weights were
    stored quantized (a ComfyUI file's layout, e.g. `int8-convrot256`); any torch
    dtype has a name; an unresolved `auto` precision names the 16-bit checkpoint's
    `compute` (a stored file's callers resolve it first)."""
    precision = "compute" if plan.precision == "auto" else plan.precision
    parts = [dtype_name(dtype), precision]
    if plan.precision in WEIGHT_ONLY:
        parts.append(f"g{plan.quant_group}")
    elif plan.precision in COMPUTE:
        parts.append(recipe_of(plan.precision, plan.fp8_recipe))
    if layout:
        parts.append(f"src:{layout}")
    return "/".join(parts)


def cache_specs(
    vae_dtype: torch.dtype, plan: TextEncoderPlan, text_dtype, layout: str = ""
) -> dict:
    """The data knobs naming both specs (`LATENT_ENCODER_SPEC`, `TEXT_ENCODER_SPEC`), to
    merge into the knobs `DataConfig.from_knobs` reads."""
    return {
        "LATENT_ENCODER_SPEC": vae_spec(vae_dtype),
        "TEXT_ENCODER_SPEC": text_spec(plan, text_dtype, layout),
    }


def _params(factory: str):
    return inspect.signature(import_object(factory)).parameters


def latent_encoder_kwargs(k: dict) -> dict:
    """The latent cache pass's IMAGE_ENCODER kwargs: IMAGE_ENCODER_KWARGS plus the
    training knobs VAE_DTYPE as `dtype` and VAE_ENCODE as `plan` where the factory takes
    them and the kwargs do not set them."""
    kwargs = dict(k["IMAGE_ENCODER_KWARGS"])
    params = _params(k["IMAGE_ENCODER"])
    if "dtype" in params:
        kwargs.setdefault("dtype", k.get("VAE_DTYPE", "auto"))
    if "plan" in params:
        kwargs.setdefault("plan", dict(k.get("VAE_ENCODE") or {}))
    return kwargs


def text_encoder_kwargs(k: dict) -> dict:
    """The text cache pass's TEXT_ENCODER kwargs: TEXT_ENCODER_KWARGS with its `plan`
    taking the training knob TEXT_ENCODE's compute fields (TEXT_SPEC_FIELDS, what
    TEXT_ENCODER_SPEC records) where the kwargs do not set them, for a factory taking a
    plan."""
    kwargs = dict(k["TEXT_ENCODER_KWARGS"])
    if "plan" not in _params(k["TEXT_ENCODER"]):
        return kwargs
    train = dict(k.get("TEXT_ENCODE") or {})
    own = dict(kwargs.get("plan") or {})
    if "dtype" in kwargs:
        own.setdefault("dtype", kwargs["dtype"])
    fill = {name: train[name] for name in TEXT_SPEC_FIELDS if name in train}
    kwargs["plan"] = {**fill, **own}
    return kwargs


class SpecEncoder:
    """An encode callable with the compute spec of what it returns (`encoder_spec`)."""

    def __init__(self, fn, spec: str):
        self.fn = fn
        self.encoder_spec = spec

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
