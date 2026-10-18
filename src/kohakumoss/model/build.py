"""Model construction: config resolution, meta-device build, parameter dtypes, op binding,
weightless (stride-0) backbones for recording op shapes."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.numerics.keep_fp32 import cast_params
from kohakumoss.model.opset import OpSet
from kohakumoss.types.arch import PRESETS, MossArchConfig
from kohakumoss.types.plans import PrecisionConfig

DEFAULT_KEEP_FP32 = PrecisionConfig().keep_fp32


def resolve_config(
    variant: str | MossArchConfig = "base", overrides: dict | None = None
) -> MossArchConfig:
    """Preset name or config, with field overrides applied."""
    cfg = PRESETS[variant] if isinstance(variant, str) else variant
    return cfg.override(**overrides) if overrides else cfg


def build_backbone(
    cfg: MossArchConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    keep_fp32: tuple[str, ...] = DEFAULT_KEEP_FP32,
    ops: OpSet | None = None,
    kernels: dict[str, str] | None = None,
    init: bool = False,
    seed: int = 0,
) -> MossBackbone:
    """Empty (meta) model when `device == "meta"`; random-initialised when `init`.

    Ops are resolved for (device, dtype) unless `ops` is given, and bound before return.
    """
    model = MossBackbone(cfg, device=device, dtype=dtype)
    if init:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.init_weights()
    cast_params(model, dtype, keep_fp32)
    if ops is None and str(device) != "meta":
        ops = OpSet.resolve(device, dtype, kernels)
    if ops is not None:
        model.bind_ops(ops)
    return model


def weightless(t: Tensor, device: torch.device | str, dtype=None) -> Tensor:
    """A stride-0 view of one zero on `device` with `t`'s shape and dtype (or
    `dtype`): one element of memory."""
    return torch.zeros((), dtype=dtype or t.dtype, device=device).expand(t.shape)


def _named(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def materialize(
    model: nn.Module, device: torch.device | str, like: dict[str, Tensor] | None = None
) -> None:
    """Every meta parameter and buffer of `model` -> `weightless` on `device`; a tensor
    of `like` with the same name and shape gives the dtype and the grad flag. Other
    buffers move to `device`; modules that derive a buffer (RopeBands) rebuild it."""
    like = like or {}
    for prefix, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.is_meta:
                continue
            ref = like.get(_named(prefix, name))
            ref = ref if ref is not None and ref.shape == p.shape else p
            view = weightless(p, device, ref.dtype)
            module._parameters[name] = nn.Parameter(view, ref.requires_grad)
        if module._buffers:
            for name, b in module.named_buffers(recurse=False):
                ref = like.get(_named(prefix, name))
                ref = ref if ref is not None and ref.shape == b.shape else b
                module._buffers[name] = (
                    weightless(b, device, ref.dtype) if b.is_meta else b.to(device)
                )
            module._apply(lambda t: t, recurse=False)


def build_shapes(
    cfg: MossArchConfig,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    trainable: bool = False,
    keep_fp32: tuple[str, ...] = DEFAULT_KEEP_FP32,
    kernels: dict[str, str] | None = None,
    prepare: Callable[[MossBackbone], None] | None = None,
    like: dict[str, Tensor] | None = None,
) -> MossBackbone:
    """A backbone holding no weight memory, for recording op shapes (the kernel
    prewarm, `kohakumoss-tune`): built on the meta device in its storage dtypes
    (`requires_grad=trainable`), `prepare(model)` applied there (base storage swaps),
    then `materialize`d on `device` (`like`: dtypes and grad flags by name), ops
    bound. Its outputs are meaningless; only the call layouts are."""
    model = MossBackbone(cfg, device="meta", dtype=dtype)
    cast_params(model, dtype, keep_fp32)
    model.requires_grad_(trainable)
    if prepare is not None:
        prepare(model)
    materialize(model, device, like)
    model.bind_ops(OpSet.resolve(device, dtype, kernels))
    return model
