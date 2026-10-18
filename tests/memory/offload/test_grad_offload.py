"""Gradient offload: the host AdamW stepping from host fp32 gradient sums matches AdamW."""

import pytest
import torch

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.grad import GradOffload
from tests._util.devices import DEVICES


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
