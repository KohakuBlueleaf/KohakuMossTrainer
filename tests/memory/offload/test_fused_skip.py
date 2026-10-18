"""Fused step with a non-finite gradient: that parameter and its optimiser state are
left untouched (a later finite step equals a fresh optimiser's first step), the other
parameters are updated. Covers each optimiser path of the fused step: the offload
AdamWs (found_inf), torch AdamW (fused on CUDA, snapshot elsewhere), AdamW16 / Muon
(`step_masked`) and bitsandbytes (snapshot and host-state restore)."""

import importlib.util

import pytest
import torch

from kohakumoss.training.optim.bundle.build import build_bundle
from kohakumoss.types.plans import PrecisionConfig
from tests._util.devices import DEVICES
from tests._util.negative import must_fail


def _optional(name: str, package: str):
    missing = importlib.util.find_spec(package) is None
    return pytest.param(
        name, marks=pytest.mark.skipif(missing, reason=f"{package} not installed")
    )


OPTIMIZERS = [
    "master_adamw",
    "host_state_adamw",
    "cpu_adamw",
    "adamw",
    "adamw16",
    "muon",
    _optional("adamw8bit", "bitsandbytes"),
]
CUDA_ONLY = ("adamw8bit",)
GROUP = {"muon": {"use_muon": True}}
HYPER = {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8}


def _bundle(name, params, device):
    groups = [
        {"params": params, "lr": 1e-2, "weight_decay": 0.1, **GROUP.get(name, {})}
    ]
    prec = PrecisionConfig(compute=torch.float32, scaler=False)
    return build_bundle(
        groups,
        torch.device(device),
        name,
        HYPER,
        {"mode": "constant"},
        10,
        prec,
        0.0,
        fused_step=True,
    )


def _params(device):
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(8, 4, device=device)) for _ in range(2)]


def _run(name, device, honour=True):
    """Step 1: p0 non-finite, p1 finite. Step 2: both finite. Returns both params."""
    params = _params(device)
    bundle = _bundle(name, params, device)
    if not honour:
        transform = bundle.transform
        bundle.transform = lambda g: transform(g) * 0
    g = torch.ones(8, 4, device=device)
    for step, bad in ((1, True), (2, False)):
        params[0].grad = g * float("inf") if bad else g.clone()
        params[1].grad = g.clone()
        for p in params:
            bundle.step_param(p)
        bundle.step()
    return params


def _fresh_first_step(name, device):
    params = _params(device)
    bundle = _bundle(name, params, device)
    params[0].grad = torch.ones(8, 4, device=device)
    bundle.step_param(params[0])
    bundle.step()
    return params[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_nonfinite_parameter_is_skipped(name, device):
    if name in CUDA_ONLY and torch.device(device).type != "cuda":
        pytest.skip(f"{name} runs on CUDA only")
    want = _fresh_first_step(name, device).detach().cpu()

    def check(honour: bool):
        got = _run(name, device, honour)[0].detach().cpu()
        assert torch.allclose(got, want, atol=1e-6)

    check(True)
    must_fail(lambda: check(False), "zeroed gradient still stepped (decay, momentum)")
