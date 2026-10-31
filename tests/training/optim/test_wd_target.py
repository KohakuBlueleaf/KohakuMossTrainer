"""WD_TARGET='start': decayed parameters decay towards their anchors (p -= lr*wd*(p-a))
in the plain bundle, the gated pull (deferred step) and the fused per-parameter post;
only decayed parameters are anchored; the knob checks (model-free: explicit parameters
and closed-form losses)."""

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.precision import build_precision
from kohakumoss.training.build.anchors import wd_target_problems
from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.optim.bundle.base import anchor_pull
from kohakumoss.training.optim.bundle.fused import _decay_to_anchor
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]

LR, WD = 1e-2, 0.5
SCHED = {"mode": "constant", "end": -1}


def _named(seed: int = 0) -> list[tuple[str, torch.nn.Parameter]]:
    g = torch.Generator().manual_seed(seed)
    return [
        ("mid.weight", torch.nn.Parameter(torch.randn(3, 4, generator=g))),
        ("mid.bias", torch.nn.Parameter(torch.randn(3, generator=g))),
    ]


def _anchors(named) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(9)
    return {n: torch.randn(p.shape, generator=g).bfloat16() for n, p in named}


def _stepped(anchored: bool) -> list[torch.Tensor]:
    named = _named()
    anchors = _anchors(named) if anchored else None
    groups = build_param_groups(named, LR, WD, "moss", anchors=anchors)
    bundle = build_bundle(
        groups,
        torch.device("cpu"),
        "adamw",
        {"lr": LR, "betas": (0.9, 0.95), "eps": 1e-8},
        SCHED,
        10,
        build_precision("bf16", ()),
        0.0,
    )
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(1))
    bundle.backward(F.linear(x, named[0][1], named[1][1]).square().mean())
    bundle.step()
    return [p.detach().clone() for _, p in named]


def check_pull(anchored_run) -> None:
    """The anchored step minus the stock step is lr * wd * a on the decayed weight and
    zero on the undecayed bias."""
    stock = _stepped(False)
    anchored = anchored_run()
    a = _anchors(_named())["mid.weight"].float()
    assert torch.allclose(anchored[0] - stock[0], LR * WD * a, atol=1e-6)
    assert torch.equal(anchored[1], stock[1])


def test_plain_bundle_pulls_towards_anchor():
    check_pull(lambda: _stepped(True))
    must_fail(lambda: check_pull(lambda: _stepped(False)), "anchors dropped")


def test_only_decayed_parameters_are_anchored():
    named = _named()
    groups = build_param_groups(named, LR, WD, "moss", anchors=_anchors(named))
    anchored = {n for g in groups if g.get("anchor") for n in g["names"]}
    assert anchored == {"mid.weight"}
    assert all(len(g["anchor"]) == len(g["params"]) for g in groups if "anchor" in g)


def test_gated_pull_skips_bad_steps():
    w = torch.nn.Parameter(torch.full((4,), 3.0))
    a = torch.full((4,), 1.0).bfloat16()
    opt = torch.optim.SGD([w], lr=LR)
    groups = [{"params": [w], "weight_decay": WD, "anchor": [a]}]
    pull = anchor_pull(groups, opt, gated=True)
    pull(torch.ones(()))
    assert torch.equal(w, torch.full((4,), 3.0))
    pull(torch.zeros(()))
    assert torch.allclose(w, torch.full((4,), 3.0 + LR * WD))


def test_fused_post_honours_found():
    p = torch.full((4,), 2.0)
    a = torch.full((4,), 4.0).bfloat16()
    _decay_to_anchor(WD, a, p, LR, torch.ones(()))
    assert torch.equal(p, torch.full((4,), 2.0))
    _decay_to_anchor(WD, a, p, LR, torch.zeros(()))
    assert torch.allclose(p, torch.full((4,), 2.0 + LR * WD * 4.0))


def test_wd_target_problems():
    k = {"WD_TARGET": "start", "WEIGHT_DECAY": 0.05, "OPTIMIZER": "adamw"}
    assert wd_target_problems(k, full=True) == []
    assert wd_target_problems(k, full=False)
    assert wd_target_problems({**k, "WEIGHT_DECAY": 0.0}, full=True)
    assert wd_target_problems({**k, "OPTIMIZER": "prodigy"}, full=True)
    assert wd_target_problems({**k, "WD_TARGET": "init"}, full=True)
    assert wd_target_problems({**k, "WD_TARGET": "zero"}, full=False) == []
