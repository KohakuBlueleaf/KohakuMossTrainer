"""Per-group hyperparameters survive every optimiser path.

Each GPU / host-state / host AdamW and the per-parameter optimizer must reproduce
`torch.optim.AdamW` run with the same groups (different lr, weight decay, betas, eps per
group). The check must fail when the groups collapse into one.
"""

import pytest
import torch

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.fused_step import PerParamOptimizer
from kohakumoss.offload.optim.gpu_adamw import HostStateAdamW, MasterAdamW
from tests._util.devices import DEVICES
from tests._util.negative import must_fail

HP = [
    dict(lr=1e-2, weight_decay=0.1),
    dict(lr=2e-2, weight_decay=0.0, betas=(0.8, 0.95)),
    dict(lr=5e-3, weight_decay=0.05, eps=1e-6),
]
STEPS = 3


def _collapse(groups: list[dict]) -> list[dict]:
    """One group with the largest weight decay (the failure mode under test)."""
    wd = max(g.get("weight_decay", 0.01) for g in groups)
    params = [p for g in groups for p in g["params"]]
    return [{"params": params, "lr": groups[0]["lr"], "weight_decay": wd}]


def _tensors(device):
    torch.manual_seed(0)
    shapes = [(16, 8), (8,), (12, 4), (4,), (6, 6)]
    return [torch.randn(s, device=device) for s in shapes]


def _grads(device):
    g = torch.Generator().manual_seed(1)
    shapes = [(16, 8), (8,), (12, 4), (4,), (6, 6)]
    return [
        [torch.randn(s, generator=g).to(device) for s in shapes] for _ in range(STEPS)
    ]


def _groups(params) -> list[dict]:
    return [
        {"params": [params[0], params[4]], **HP[0]},
        {"params": [params[1], params[3]], **HP[1]},
        {"params": [params[2]], **HP[2]},
    ]


def _run(device, make, collapse=False, per_param=False):
    params = [torch.nn.Parameter(t) for t in _tensors(device)]
    groups = _groups(params)
    opt = make(_collapse(groups) if collapse else groups)
    for grads in _grads(device):
        for p, g in zip(params, grads):
            p.grad = g.clone()
        if per_param:
            for p in reversed(params):
                opt.step_param(p)
            if hasattr(opt, "finish"):
                opt.finish()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
    if hasattr(opt, "synchronize"):
        opt.synchronize()
    return [p.detach().float().cpu() for p in params]


def _err(got, want) -> float:
    return max((a - b).abs().max().item() for a, b in zip(got, want))


OPTIMS = {
    "master": MasterAdamW,
    "host_state": HostStateAdamW,
    "cpu_adamw": CPUAdamW,
    "per_param": lambda groups: PerParamOptimizer(groups, torch.optim.AdamW),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("per_param", [False, True])
@pytest.mark.parametrize("name", list(OPTIMS))
def test_optimizer_keeps_group_hyperparameters(device, name, per_param):
    want = _run(device, torch.optim.AdamW)
    got = _run(device, OPTIMS[name], per_param=per_param)
    assert _err(got, want) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_group_check_catches_collapsed_groups(device):
    want = _run(device, torch.optim.AdamW)

    def check():
        assert _err(_run(device, MasterAdamW, collapse=True), want) < 1e-5

    must_fail(check, "groups collapsed into one (largest weight decay)")


@pytest.mark.cpu
def test_fused_per_param_groups_track_schedule_edits():
    params = [torch.nn.Parameter(t) for t in _tensors("cpu")]
    per = PerParamOptimizer(_groups(params), torch.optim.SGD, lr=1.0)
    per.param_groups[2]["lr"] = 0.0
    before = params[2].detach().clone()
    params[2].grad = torch.ones_like(params[2])
    per.step_param(params[2])
    assert torch.equal(params[2].detach(), before)
