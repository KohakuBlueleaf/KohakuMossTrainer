"""The host AdamW reproduces the stock AdamW, whole-step and per-parameter."""

import pytest
import torch

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from tests._util.devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
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
    host.finish()
    assert host.state[b]["step"].item() == 6
