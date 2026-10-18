"""Optimiser registry: name -> builder(groups, device, **hyper) -> torch Optimizer.
OPTIMIZER names an algorithm and implementation; the placement builders of `adamw`
(INTERNAL) are chosen by OPTIM_STATE, never named by OPTIMIZER."""

import torch

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW, MasterAdamW
from kohakumoss.training.optim.optimizers.adamw8bit import AdamW8bitKernel
from kohakumoss.training.optim.optimizers.adamw16 import AdamW16
from kohakumoss.training.optim.optimizers.muon import MuonHybrid
from kohakumoss_common.registry import Registry

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
try:
    import prodigyopt
except ImportError:
    prodigyopt = None
try:
    import schedulefree
except ImportError:
    schedulefree = None

OPTIMIZERS = Registry("optimizer")
OPTIMIZER_SPECIAL_KEYS = ("names", "splits", "decay_to_one", "decay_to_one_coef")
INTERNAL = {"cpu_adamw": "cpu", "host_state_adamw": "stream", "master_adamw": "gpu"}
STATE_BYTES = {
    "adamw": 8,
    "adamw16": 4,
    "adamw8bit": 2,
    "adamw8bit_kernel": 2,
    "muon": 8,
    "prodigy": 16,
    "schedule_free": 8,
}


def _plain_groups(groups: list[dict]) -> list[dict]:
    """Strip KohakuMoss-only group keys that stock optimisers reject."""
    return [
        {k: v for k, v in g.items() if k not in OPTIMIZER_SPECIAL_KEYS + ("use_muon",)}
        for g in groups
    ]


def master_groups(groups: list[dict]) -> list[dict]:
    """Groups for the offload AdamWs, which apply `decay_to_one` themselves."""
    return [
        {k: v for k, v in g.items() if k not in ("names", "splits", "use_muon")}
        for g in groups
    ]


def _need(module, name: str):
    if module is None:
        raise SystemExit(f"OPTIMIZER={name!r} needs the `{name}` package installed")
    return module


@OPTIMIZERS.register("adamw")
def build_adamw(groups, device, lr, betas, eps, **_):
    fused = device.type == "cuda" and all(
        p.dtype in (torch.float32, torch.float16, torch.bfloat16)
        for g in groups
        for p in g["params"]
    )
    return torch.optim.AdamW(
        _plain_groups(groups), lr=lr, betas=betas, eps=eps, fused=fused or None
    )


@OPTIMIZERS.register("adamw16")
def build_adamw16(groups, device, lr, betas, eps, rounding="nearest", **_):
    return AdamW16(
        _plain_groups(groups), lr=lr, betas=betas, eps=eps, rounding=rounding
    )


@OPTIMIZERS.register("adamw8bit")
def build_adamw8bit(groups, device, lr, betas, eps, **_):
    lib = _need(bnb, "bitsandbytes")
    return lib.optim.AdamW8bit(_plain_groups(groups), lr=lr, betas=betas, eps=eps)


@OPTIMIZERS.register("adamw8bit_kernel")
def build_adamw8bit_kernel(groups, device, lr, betas, eps, rounding="nearest", **_):
    return AdamW8bitKernel(
        _plain_groups(groups), lr=lr, betas=betas, eps=eps, rounding=rounding
    )


@OPTIMIZERS.register("muon")
def build_muon(groups, device, lr, betas, eps, momentum=0.95, ns_steps=5, **_):
    keep = [{k: v for k, v in g.items() if k not in ("names",)} for g in groups]
    for g in keep:
        g.pop("decay_to_one", None)
        g.pop("decay_to_one_coef", None)
    return MuonHybrid(
        keep, lr=lr, betas=betas, eps=eps, momentum=momentum, ns_steps=ns_steps
    )


@OPTIMIZERS.register("cpu_adamw")
def build_cpu_adamw(groups, device, lr, betas, eps, pinned=True, **_):
    return CPUAdamW(master_groups(groups), lr=lr, betas=betas, eps=eps, pinned=pinned)


@OPTIMIZERS.register("master_adamw")
def build_master_adamw(groups, device, lr, betas, eps, **_):
    return MasterAdamW(
        master_groups(groups), lr=lr, betas=betas, eps=eps, weight_decay=0.0
    )


@OPTIMIZERS.register("host_state_adamw")
def build_host_state_adamw(groups, device, lr, betas, eps, pinned=True, **_):
    return HostStateAdamW(
        master_groups(groups),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=0.0,
        pinned=pinned,
    )


@OPTIMIZERS.register("prodigy")
def build_prodigy(groups, device, lr, betas, eps, **_):
    lib = _need(prodigyopt, "prodigyopt")
    return lib.Prodigy(_plain_groups(groups), lr=lr, betas=betas, eps=eps)


@OPTIMIZERS.register("schedule_free")
def build_schedule_free(groups, device, lr, betas, eps, warmup_steps=0, **_):
    lib = _need(schedulefree, "schedulefree")
    return lib.AdamWScheduleFree(
        _plain_groups(groups), lr=lr, betas=betas, eps=eps, warmup_steps=warmup_steps
    )
