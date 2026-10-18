"""OPTIMIZER='adamw8bit_kernel': the project's 8-bit AdamW steps every parameter with the
kernel library's `adamw8bit_step` over its own blockwise 8-bit states, step count from 1
(model-free: explicit parameters on the host, the eager reference)."""

import pytest
import torch

from kohakumoss.training.optim.optimizers.factory import OPTIMIZERS
from kohakumoss_kernels.optim import new_8bit_state, reference
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
HYPER = {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8}


def _grads(step: int, shape) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(step))


def _reference(steps: int, shape, wd: float) -> torch.Tensor:
    p = torch.linspace(-1, 1, shape[0] * shape[1]).view(shape)
    state = new_8bit_state(p)
    for t in range(1, steps + 1):
        reference.adamw8bit_step(
            p, _grads(t, shape), *state, None, 1e-2, 0.9, 0.95, 1e-8, wd, t
        )
    return p


def check_matches(build) -> None:
    shape = (8, 64)
    p = torch.nn.Parameter(torch.linspace(-1, 1, 8 * 64).view(shape))
    groups = [{"params": [p], "weight_decay": 0.1}]
    opt = build(groups, torch.device("cpu"), **HYPER)
    for t in range(1, 4):
        p.grad = _grads(t, shape)
        opt.step()
    torch.testing.assert_close(p.detach(), _reference(3, shape, 0.1))


def test_kernel_adamw8bit_matches_the_reference():
    check_matches(OPTIMIZERS.get("adamw8bit_kernel"))

    def dense(groups, device, lr, betas, eps):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)

    must_fail(lambda: check_matches(dense), "fp32 AdamW instead of the 8-bit states")
