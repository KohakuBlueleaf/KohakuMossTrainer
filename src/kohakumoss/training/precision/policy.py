"""Precision policy: compute dtype, fp32-kept modules, parameter dtypes, autocast context."""

import contextlib
from collections.abc import Callable
from functools import partial

import torch
from torch import nn

from kohakumoss.types.plans import PrecisionPolicy

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def build_policy(
    precision: str,
    keep_fp32: tuple[str, ...],
    fp16_fix: str = "none",
    fp16_fix_kwargs: dict | None = None,
    master: str = "fp32",
    scaler_init: float = 2.0**16,
    growth_interval: int = 2000,
) -> PrecisionPolicy:
    """PrecisionPolicy from PRECISION / KEEP_FP32 / FP16_FIX(_KWARGS); fp16 enables the
    loss scaler."""
    return PrecisionPolicy(
        compute=DTYPES[precision],
        master=DTYPES[master],
        keep_fp32=tuple(keep_fp32),
        scaler=precision == "fp16",
        scaler_init=scaler_init,
        scaler_growth_interval=growth_interval,
        fp16_fix=fp16_fix,
        fp16_fix_kwargs=dict(fp16_fix_kwargs or {}),
    )


def _kept(name: str, keep: tuple[str, ...]) -> bool:
    return any(
        name == k or name.startswith(k + ".") or f".{k}." in f".{name}" for k in keep
    )


def apply_param_dtypes(
    model: nn.Module, policy: PrecisionPolicy, trainable: set[int], master_16bit: bool
) -> None:
    """Trainable params -> master dtype (compute dtype for AdamW16, fp32 for KEEP_FP32
    modules). Frozen params keep the dtype the loader / memory plan gave them."""
    for name, p in model.named_parameters():
        if id(p) not in trainable:
            continue
        if _kept(name, policy.keep_fp32):
            dtype = torch.float32
        else:
            dtype = policy.compute if master_16bit else policy.master
        if p.dtype != dtype:
            p.data = p.data.to(dtype)


def autocast_factory(policy: PrecisionPolicy, device: torch.device) -> Callable:
    """Zero-arg context factory: autocast to the compute dtype, or a null context in fp32."""
    if policy.compute == torch.float32:
        return contextlib.nullcontext
    return partial(torch.autocast, device_type=device.type, dtype=policy.compute)
