"""apply(model, config): build LyCORIS modules on the preset's targets and bind forwards."""

import torch
from torch import nn

from kohakumoss.adapters.build.config import AdapterConfig
from kohakumoss.adapters.build.network import build_targets
from kohakumoss.adapters.build.target import CHILD, Target, base_kind
from kohakumoss.adapters.lycoris.backend import pick_backend
from kohakumoss.adapters.train.adapter_set import LycoAdapterSet
from kohakumoss.logging.log import get_logger

log = get_logger(__name__)


def compute_device(model: nn.Module) -> torch.device:
    """Device of the first non-empty parameter (base storage may sit elsewhere)."""
    for p in model.parameters():
        if p.numel() > 0:
            return p.device
    return torch.device("cpu")


def _place(t: Target, dtype: torch.dtype, device: torch.device) -> None:
    """Register the module under its linear (FSDP / offload see it) on `device`."""
    t.lyco.to(device=device, dtype=dtype)
    t.lyco.train(t.linear.training)
    t.linear.add_module(CHILD, t.lyco)


def note_bypassed(config: AdapterConfig, targets: list[Target]) -> None:
    """ADAPTER_MODE='merge' on layers whose base is not a plain 16/32-bit dense weight:
    they run bypass (a WARNING names the effective mode and the layers)."""
    if config.mode != "merge":
        return
    stored = [f"{t.path} ({base_kind(t.linear)})" for t in targets if t.stored]
    if not stored:
        return
    log.warning(
        f"ADAPTER_MODE='merge': {len(stored)} of {len(targets)} layers have a "
        f"quantized / streamed / fp8-compute base and run as bypass, e.g. {stored[:4]}; "
        f"the other {len(targets) - len(stored)} merge"
    )


def apply(
    model: nn.Module,
    config: AdapterConfig,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> LycoAdapterSet:
    """Adapters on `model` per `config`; parameters live in `dtype` (the master dtype)."""
    problems = config.validate()
    if problems:
        raise ValueError("; ".join(problems))
    device = torch.device(device) if device is not None else compute_device(model)
    backend = pick_backend(device, config.backend)
    targets, norms = build_targets(model, config)
    if not targets and not norms:
        raise ValueError(f"preset {config.preset!r} matched no layer")
    note_bypassed(config, targets)
    for t in targets:
        _place(t, dtype, device)
    return LycoAdapterSet(model, config, targets, norms, backend)
