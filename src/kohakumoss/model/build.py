"""Model construction: config resolution, meta-device build, dtype policy, op binding."""

import torch

from kohakumoss.model.backbone import MossBackbone
from kohakumoss.model.numerics.keep_fp32 import apply_dtype_policy
from kohakumoss.model.opset import OpSet
from kohakumoss.types.arch import PRESETS, MossArchConfig

DEFAULT_KEEP_FP32 = (
    "time_embedding",
    "shared_block_modulation",
    "video_head_modulation",
)


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
    apply_dtype_policy(model, dtype, keep_fp32)
    if ops is None and str(device) != "meta":
        ops = OpSet.resolve(device, dtype, kernels)
    if ops is not None:
        model.bind_ops(ops)
    return model


def random_backbone(
    variant: str = "tiny",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    **overrides,
) -> MossBackbone:
    """Random model of a preset (the tiny fixture for tests and smoke runs)."""
    cfg = resolve_config(variant, overrides or None)
    return build_backbone(cfg, device=device, dtype=dtype, init=True, seed=seed)
