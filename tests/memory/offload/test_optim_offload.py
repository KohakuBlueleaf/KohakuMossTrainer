"""Fused in-backward steps and the host AdamW reproduce the stock optimiser."""

import pytest
import torch

from kohakumoss.offload.cpu_optim import CPUAdamW
from kohakumoss.offload.fused_step import FusedStep, PerParamOptimizer
from tests._util.negative import must_fail
from tests.parallel._common import batch, make_model


def _loss(model, seed):
    img, txt, sigma, seq, target = batch(seed=seed)
    return (model(img, txt, sigma, seq) - target).square().mean()


def _train_ref(steps, accum, opt_cls, **kw):
    model = make_model()
    opt = opt_cls(model.parameters(), **kw)
    for s in range(steps):
        for a in range(accum):
            _loss(model, 10 * s + a).backward()
        opt.step()
        opt.zero_grad()
    return model


def _train_fused(steps, accum, opt_cls, armed=True, **kw):
    model = make_model()
    per = PerParamOptimizer(model.parameters(), opt_cls, **kw)
    fused = FusedStep(model.parameters(), per.step_param)
    for s in range(steps):
        for a in range(accum):
            fused.arm(armed and a == accum - 1)
            _loss(model, 10 * s + a).backward()
    return model


def _diff(a, b):
    return max(
        (p - q).abs().max().item() for p, q in zip(a.parameters(), b.parameters())
    )


@pytest.mark.cpu
@pytest.mark.parametrize("opt", [torch.optim.SGD, torch.optim.AdamW])
def test_fused_step_matches_optimizer(opt):
    kw = {"lr": 1e-2}
    ref = _train_ref(3, 2, opt, **kw)
    got = _train_fused(3, 2, opt, **kw)
    assert _diff(ref, got) < 1e-10


@pytest.mark.cpu
def test_fused_step_check_catches_a_hook_that_never_steps():
    ref = _train_ref(2, 1, torch.optim.SGD, lr=1e-2)

    def check():
        assert (
            _diff(ref, _train_fused(2, 1, torch.optim.SGD, armed=False, lr=1e-2))
            < 1e-10
        )

    must_fail(check, "unarmed fused step")


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_cpu_adamw_matches_adamw(device):
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(64, 32, device=device))
    b = torch.nn.Parameter(a.detach().clone())
    ref = torch.optim.AdamW([a], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    host = CPUAdamW([b], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    for s in range(5):
        g = torch.randn_like(a)
        a.grad, b.grad = g.clone(), g.clone()
        ref.step()
        host.step()
    assert (a - b).abs().max().item() < 1e-5
    b.grad = torch.randn_like(b)
    host.step_param(b)
    assert host.state[b]["step"].item() == 6
