"""OptimBundle: loss scaling, skip on non-finite, clipping, decay-to-one, scale schedule
(model-free: explicit parameter lists and closed-form losses)."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.precision import DynamicScaler, build_precision
from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.optim.bundle.base import decay_to_one
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

SCHED = {"mode": "constant", "end": -1}


def _params(
    n_in: int, n_out: int, seed: int = 0
) -> list[tuple[str, torch.nn.Parameter]]:
    """(name, parameter) of one affine map y = x W^T + b."""
    g = torch.Generator().manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(n_out, n_in, generator=g) / n_in**0.5)
    b = torch.nn.Parameter(torch.randn(n_out, generator=g) * 0.1)
    return [("weight", w), ("bias", b)]


def _affine(named, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, named[0][1], named[1][1])


def _bundle(named, precision="fp16", clip=0.0, lr=1e-2):
    groups = build_param_groups(named, lr, 0.0, "none")
    hyper = {"lr": lr, "betas": (0.9, 0.95), "eps": 1e-8}
    return build_bundle(
        groups,
        torch.device("cpu"),
        "adamw",
        hyper,
        SCHED,
        100,
        build_precision(precision, ()),
        clip,
    )


def test_scaled_grads_are_unscaled_before_step():
    a, b = _params(4, 3), _params(4, 3)
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    bundle = _bundle(a, "fp16")
    bundle.backward(_affine(a, x).square().mean())
    grads = [p.grad.clone() / bundle.scaler.scale for _, p in a]
    _affine(b, x).square().mean().backward()
    for g, (_, p) in zip(grads, b):
        assert torch.allclose(g, p.grad, rtol=1e-5)
    assert bundle.step() is True


def test_nonfinite_skips_and_backs_off():
    named = _params(4, 3)
    before = [p.detach().clone() for _, p in named]
    bundle = _bundle(named, "fp16")
    scale0 = bundle.scaler.scale
    bundle.backward(_affine(named, torch.full((2, 4), float("inf"))).sum())
    assert bundle.step() is False
    assert bundle.scaler.scale == scale0 / 2 and bundle.skipped == 1
    for (_, p), q in zip(named, before):
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
    named = _params(8, 8)
    bundle = _bundle(named, "bf16", clip=0.5)
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(2)) * 100
    bundle.backward(_affine(named, x).square().sum())
    grads = [p.grad for _, p in named]
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


def _moss_named() -> list[tuple[str, torch.nn.Parameter]]:
    """The names of an embedder, a hidden linear and a norm, as explicit tensors."""
    shapes = {
        "video_embedder.weight": (4, 4),
        "video_embedder.bias": (4,),
        "mid.weight": (4, 4),
        "mid.bias": (4,),
        "norm.weight": (4,),
        "norm.bias": (4,),
    }
    return [(n, torch.nn.Parameter(torch.ones(s))) for n, s in shapes.items()]


def _wd_sets(rules: str) -> tuple[set, set]:
    groups = build_param_groups(
        _moss_named(), 1e-3, 0.01, rules, decay_to_one=r"^norm\.weight$"
    )
    decayed = {n for g in groups if g["weight_decay"] > 0 for n in g["names"]}
    to_one = {n for g in groups if g["decay_to_one"] for n in g["names"]}
    return decayed, to_one


def _moss_sets(sets: tuple[set, set]) -> None:
    assert sets == ({"mid.weight"}, {"norm.weight"}), sets


def test_moss_wd_rules():
    _moss_sets(_wd_sets("moss"))
    must_fail(lambda: _moss_sets(_wd_sets("all")), "decay on every parameter")
