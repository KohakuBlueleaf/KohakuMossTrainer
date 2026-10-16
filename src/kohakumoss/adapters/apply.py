"""apply(model, spec): build LyCORIS modules on the preset's targets and bind forwards."""

import torch
from lycoris.kernels import set_compile_dynamic
from torch import nn

from kohakumoss.adapters.adapter_set import LycoAdapterSet
from kohakumoss.adapters.backend import pick_backend
from kohakumoss.adapters.network import build_targets
from kohakumoss.adapters.spec import AdapterSpec
from kohakumoss.adapters.target import CHILD, Target


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
    if t.kind == "full":
        t.lyco.apply_to()


def apply(
    model: nn.Module,
    spec: AdapterSpec,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> LycoAdapterSet:
    """Adapters on `model` per `spec`; parameters live in `dtype` (the master dtype)."""
    problems = spec.validate()
    if problems:
        raise ValueError("; ".join(problems))
    targets, norms = build_targets(model, spec)
    if not targets and not norms:
        raise ValueError(f"preset {spec.preset!r} matched no layer")
    stored_full = [t.path for t in targets if t.kind == "full" and t.stored]
    if stored_full:
        raise ValueError(f"ALGO='full' needs a dense base weight: {stored_full[:4]}")
    device = torch.device(device) if device is not None else compute_device(model)
    for t in targets:
        _place(t, dtype, device)
    set_compile_dynamic(spec.compile_dynamic)
    backend = pick_backend(device, spec.backend)
    return LycoAdapterSet(model, spec, targets, norms, backend)
