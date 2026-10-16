"""OptimBundle: loss scaling, skip on non-finite, clipping, decay-to-one, scale schedule."""

import pytest
import torch

from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.optim.bundle import decay_to_one
from kohakumoss.training.precision import DynamicScaler, build_policy
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

SCHED = {"mode": "constant", "end": -1}


def _bundle(model, precision="fp16", clip=0.0, lr=1e-2):
    groups = build_param_groups(model.named_parameters(), lr, 0.0, "none")
    hyper = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    return build_bundle(
        groups,
        torch.device("cpu"),
        "adamw",
        hyper,
        SCHED,
        100,
        build_policy(precision, ()),
        clip,
    )


def test_scaled_grads_are_unscaled_before_step():
    torch.manual_seed(0)
    a, b = torch.nn.Linear(4, 3), torch.nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    x = torch.randn(5, 4)
    bundle = _bundle(a, "fp16")
    bundle.backward(a(x).square().mean())
    grads = [p.grad.clone() / bundle.scaler.scale for p in a.parameters()]
    b(x).square().mean().backward()
    for g, p in zip(grads, b.parameters()):
        assert torch.allclose(g, p.grad, rtol=1e-5)
    assert bundle.step() is True


def test_nonfinite_skips_and_backs_off():
    model = torch.nn.Linear(4, 3)
    before = [p.detach().clone() for p in model.parameters()]
    bundle = _bundle(model, "fp16")
    scale0 = bundle.scaler.scale
    bundle.backward(model(torch.full((2, 4), float("inf"))).sum())
    assert bundle.step() is False
    assert bundle.scaler.scale == scale0 / 2 and bundle.skipped == 1
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p, q)


def test_scaler_floor_and_growth():
    s = DynamicScaler(init=2.0, growth_interval=3)
    for _ in range(5):
        s.update(True)
    assert s.scale == 1.0
    for _ in range(3):
        s.update(False)
    assert s.scale == 2.0


def test_clip_to_max_norm():
    model = torch.nn.Linear(8, 8)
    bundle = _bundle(model, "bf16", clip=0.5)
    bundle.backward(model(torch.randn(4, 8) * 100).square().sum())
    grads = [p.grad for p in model.parameters()]
    bundle.clip(grads, bundle.grad_norm(grads))
    total = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    assert abs(total.item() - 0.5) < 1e-3


def test_decay_to_one_pulls_towards_one():
    w = torch.nn.Parameter(torch.full((4,), 3.0))
    opt = torch.optim.SGD([w], lr=0.1)
    groups = [{"params": [w], "decay_to_one": True, "decay_to_one_coef": 0.5}]
    decay_to_one(groups, opt)()
    assert torch.allclose(w, torch.full((4,), 3.0 - 0.1 * 0.5 * 2.0))

    def broken():
        w2 = torch.nn.Parameter(torch.full((4,), 3.0))
        opt2 = torch.optim.SGD([w2], lr=0.1)
        decay_to_one(
            [{"params": [w2], "decay_to_one": False, "decay_to_one_coef": 0.5}], opt2
        )()
        assert not torch.equal(w2, torch.full((4,), 3.0))

    must_fail(broken, "decay-to-one disabled")


def test_moss_wd_rules():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.video_embedder = torch.nn.Linear(4, 4)
            self.mid = torch.nn.Linear(4, 4)
            self.norm = torch.nn.LayerNorm(4)

    groups = build_param_groups(
        M().named_parameters(), 1e-3, 0.01, "moss", decay_to_one=r"^norm\.weight$"
    )
    decayed = {n for g in groups if g["weight_decay"] > 0 for n in g["names"]}
    to_one = {n for g in groups if g["decay_to_one"] for n in g["names"]}
    assert decayed == {"mid.weight"}
    assert to_one == {"norm.weight"}
