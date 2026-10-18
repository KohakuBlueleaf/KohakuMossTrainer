"""CPUAdamW's host buffers, model-free (explicit 16-bit parameters on the host): a 16-bit
gradient is converted for the update into one reused fp32 buffer (not a new tensor per
parameter and step), a lane's gradient copy exists only once a gradient went through
it, and the update equals AdamW on the fp32 master."""

import pytest
import torch

from kohakumoss.offload.optim import cpu_optim
from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _params():
    g = torch.Generator().manual_seed(0)
    return [
        torch.nn.Parameter(torch.randn(n, generator=g).bfloat16()) for n in (40, 24)
    ]


def check_reuse(opt_cls) -> None:
    params = _params()
    ref = [torch.nn.Parameter(p.detach().float()) for p in params]
    opt = opt_cls(params, lr=1e-2, weight_decay=0.0)
    want = torch.optim.AdamW(ref, lr=1e-2, weight_decay=0.0)
    assert all(lane.grad_host is None for lane in opt._lanes.values())
    held = []
    for step in range(3):
        for p, r in zip(params, ref):
            g = torch.full_like(r, 0.1 * (step + 1))
            p.grad, r.grad = g.bfloat16(), g.bfloat16().float()
        opt.step()
        want.step()
        held.append(opt._scratch["main"].flat)
    ptrs = {t.untyped_storage().data_ptr() for t in held}
    assert len(ptrs) == 1, f"{len(ptrs)} conversion buffers over 3 steps"
    for p, r in zip(params, ref):
        torch.testing.assert_close(opt.state[p]["master"], r.detach())


class PerStep(CPUAdamW):
    """The broken variant: a new fp32 scratch for every step."""

    def step(self, closure=None):
        self._scratch["main"] = cpu_optim._Scratch()
        return super().step(closure)


def test_conversion_buffer_reused():
    check_reuse(CPUAdamW)
    must_fail(lambda: check_reuse(PerStep), "a conversion per step")
