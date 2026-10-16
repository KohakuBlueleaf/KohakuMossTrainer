"""Resolved plans: precision policy, memory plan, parallel plan (built once at startup)."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    """Compute dtype, modules kept in fp32, loss scaling and the residual fold."""

    compute: torch.dtype = torch.bfloat16
    master: torch.dtype = torch.float32
    keep_fp32: tuple[str, ...] = (
        "time_embedding",
        "shared_block_modulation",
        "video_head_modulation",
    )
    scaler: bool = False
    scaler_init: float = 2.0**16
    scaler_growth_interval: int = 2000
    fold: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class MemoryPlan:
    """Concrete memory/speed knobs a profile resolves to; every field is overridable."""

    profile: str = "balanced"
    kernels: dict = field(default_factory=dict)
    checkpoint: str = "block"
    act_offload: bool = False
    base_precision: str = "compute"
    stream: str = "off"
    stream_resident: int = 0
    pinned: bool = True
    optim_state: str = "gpu"
    fused_step: bool = False
    compile: str = "off"


@dataclass(frozen=True)
class ParallelPlan:
    """Multi-device layout and process launch."""

    kind: str = "none"
    backend: str = "auto"
    init: str = "tcp"
    devices: tuple[int, ...] = (0,)
    stages: tuple[int, ...] = ()
    schedule: str = "1f1b"
    microbatches: int = 1
    token_budgets: tuple[int, ...] = ()
