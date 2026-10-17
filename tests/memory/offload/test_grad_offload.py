"""Gradient offload: no device gradients survive a micro-batch, host fp32 sums match."""

import pytest
import torch

from kohakumoss.offload.cpu_optim import CPUAdamW
from kohakumoss.offload.grad import GradOffload
from kohakumoss.training.memory.runners import build_runner
from tests._shared.tiny import DEVICES, batch, make_model
from tests._util.negative import must_fail


def _loss(model, device, seed):
    img, txt, sigma, seq, target = batch(device, seed=seed)
    return (model(img, txt, sigma, seq) - target).square().mean()


def _reference(device, accum, checkpoint="none"):
    model = make_model(device)
    model.runner = build_runner(checkpoint, model.layers)
    for a in range(accum):
        _loss(model, device, a).backward()
    return {n: p.grad.detach().double().cpu() for n, p in model.named_parameters()}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("checkpoint", ["none", "block", "sub-block", "selective"])
def test_grad_offload_accumulates_on_host(device, checkpoint):
    accum = 3
    ref = _reference(device, accum, checkpoint)
    model = make_model(device)
    model.runner = build_runner(checkpoint, model.layers)
    off = GradOffload(model.parameters())
    for a in range(accum):
        _loss(model, device, a).backward()
        assert all(p.grad is None for p in model.parameters())
    off.flush()
    names = dict(model.named_parameters())
    err = max(
        ((off.host[names[n]].double() - g).abs().max() / (g.abs().max() + 1e-30)).item()
        for n, g in ref.items()
    )
    assert err < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_check_catches_a_lost_micro_batch(device):
    ref = _reference(device, 3)

    def check():
        model = make_model(device)
        off = GradOffload(model.parameters())
        for a in range(2):
            _loss(model, device, a).backward()
        off.flush()
        p = dict(model.named_parameters())["out.weight"]
        assert (off.host[p].double() - ref["out.weight"]).abs().max() < 1e-6

    must_fail(check, "a micro-batch that never reached the host sum")


@pytest.mark.parametrize("device", DEVICES)
def test_cpu_adamw_from_host_grads_matches(device):
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(48, 40, device=device))
    b = torch.nn.Parameter(a.detach().clone())
    ref = torch.optim.AdamW([a], lr=1e-2, weight_decay=0.1)
    off = GradOffload([b])
    host = CPUAdamW([b], lr=1e-2, weight_decay=0.1)
    host.use_host_grads(off)
    for step in range(3):
        gs = [torch.randn_like(a) for _ in range(2)]
        a.grad = gs[0] + gs[1]
        ref.step()
        for g in gs:
            (b * g).sum().backward()
        off.flush()
        assert b.grad is None
        host.step()
        off.zero()
    assert (a - b).abs().max().item() < 1e-5
