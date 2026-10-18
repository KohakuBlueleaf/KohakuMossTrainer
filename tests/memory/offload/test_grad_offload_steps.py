"""Gradient offload across micro-batches and steps, model-free (explicit parameters on
the host): a step's first gradient replaces the host sum (no zeroing between steps), a
parameter without a gradient this step gets none back, and a parameter whose hook does
not fire in the last micro-batch is finished from its earlier micro-batches' sum."""

import pytest
import torch

from kohakumoss.offload.optim.cpu_optim import CPUAdamW
from kohakumoss.offload.optim.grad import GradOffload
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _params() -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(torch.zeros(3, 4)) for _ in range(2)]


def _backward(params, coef: float, which=(0, 1)) -> None:
    sum((params[i] * coef).sum() for i in which).backward()


def check_steps(off: GradOffload, params) -> None:
    """Step 1 sums two micro-batches, step 2 one; parameter 1 has none in step 2."""
    for _ in range(2):
        off.start(False)
        _backward(params, 1.0)
    off.flush()
    assert torch.equal(off.host[params[0]], torch.full((3, 4), 2.0))
    off.zero()
    assert torch.equal(off.host[params[0]], torch.full((3, 4), 2.0)), "sum zeroed"
    off.start(False)
    _backward(params, 3.0, which=(0,))
    off.flush()
    assert torch.equal(off.host[params[0]], torch.full((3, 4), 3.0))
    assert off.filled_sums() == [off.host[params[0]]]
    off.restore(params[1])
    assert params[1].grad is None, "a stale sum came back as a gradient"


def test_first_gradient_replaces_the_sum(monkeypatch):
    params = _params()
    check_steps(GradOffload(params, pinned=False), params)
    monkeypatch.setattr(GradOffload, "_offload_host", _always_add)
    params = _params()
    adding = GradOffload(params, pinned=False)
    must_fail(lambda: check_steps(adding, params), "old sums kept under new ones")


def _always_add(self, lane, p) -> None:
    lane.host.add_(p.grad)
    lane.filled = True
    p.grad = None


def check_settled(off: GradOffload, params) -> dict:
    """Parameter 1 has gradients in micro-batch 1 only; the final callback must still
    see its sum once the last backward is done."""
    got = {}

    def final(p):
        got[id(p)] = p.grad.clone()

    off.start(False, final)
    _backward(params, 2.0)
    off.start(True, final)
    _backward(params, 5.0, which=(0,))
    off.finish()
    assert torch.equal(got[id(params[0])], torch.full((3, 4), 7.0)), got
    assert id(params[1]) in got, "parameter 1 never finished"
    assert torch.equal(got[id(params[1])], torch.full((3, 4), 2.0))
    return got


def test_late_parameter_is_settled(monkeypatch):
    params = _params()
    check_settled(GradOffload(params, pinned=False), params)
    params = _params()
    lazy = GradOffload(params, pinned=False)
    monkeypatch.setattr(lazy, "settle", lambda: None)
    must_fail(lambda: check_settled(lazy, params), "settle skipped")


def test_host_adamw_steps_only_filled_sums():
    params = _params()
    off = GradOffload(params, pinned=False)
    opt = CPUAdamW(params, lr=1e-1, weight_decay=0.0)
    opt.use_host_grads(off)
    off.start(False)
    _backward(params, 1.0, which=(0,))
    off.flush()
    before = params[1].detach().clone()
    opt.step()
    assert torch.equal(params[1].detach(), before), "a parameter with no sum moved"
    assert not torch.equal(params[0].detach(), torch.zeros(3, 4))
    assert opt._lanes[id(params[1])].grad_host is off.host[params[1]]
