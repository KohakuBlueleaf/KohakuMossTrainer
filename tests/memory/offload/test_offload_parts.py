"""Gradient offload with a host optimiser: the bundle's gradient list (unscale, norm,
clip) holds the offloaded sums of this step and the streamed blocks' host gradients;
a sum with no gradient this step is left out."""

import pytest
import torch

from kohakumoss.offload.optim.grad import GradOffload
from kohakumoss.training.memory import optimizers
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _parts(masters):
    used, unused = torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(2))
    grads = GradOffload([used, unused], pinned=False)
    grads.start(False)
    (used * 3.0).sum().backward()
    grads.drain()
    by_id = {id(used): "used", id(unused): "unused"}
    by_id.update({id(m): f"master{i}" for i, m in enumerate(masters)})
    parts = optimizers._offload_parts(grads, True, False, masters, by_id)
    return parts, grads, used, unused


def test_cpu_adamw_grads_include_streamed_masters(monkeypatch):
    master = torch.nn.Parameter(torch.ones(3))
    master.grad = torch.full((3,), 2.0)

    def check():
        parts, grads, used, unused = _parts([master])
        ids = {id(g) for g in parts["grads"]()}
        assert id(master.grad) in ids
        assert id(grads.host[used]) in ids
        assert id(grads.host[unused]) not in ids, "a sum with no gradient this step"
        assert parts["grad_names"]() == ["used", "master0"], parts["grad_names"]()

    check()
    monkeypatch.setattr(
        optimizers, "_host_grads", lambda grads, masters: grads.filled_sums()
    )
    must_fail(check, "the offloaded lanes alone")
    monkeypatch.setattr(
        optimizers, "_host_grads", lambda grads, masters: list(grads.host.values())
    )
    must_fail(check, "every sum, filled or not")
