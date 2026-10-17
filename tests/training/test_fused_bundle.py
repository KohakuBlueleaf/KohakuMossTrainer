"""FusedStepBundle against the plain Bundle: same update, same decay-to-one, the real
gradient norm, and honest accounting of a step with a non-finite gradient (its
parameters are stepped inside backward, so it is applied, not skipped)."""

import pytest
import torch

from kohakumoss.training.optim import build_bundle, build_param_groups
from kohakumoss.training.precision import build_policy
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SCHED = {"mode": "constant", "end": -1}


def _model(seed: int = 0):
    torch.manual_seed(seed)
    m = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.LayerNorm(5))
    with torch.no_grad():
        m[1].weight.fill_(3.0)
    return m


def _bundle(model, fused: bool, precision="fp32"):
    groups = build_param_groups(
        model.named_parameters(), 1e-2, 0.5, "moss", decay_to_one=r"^1\.weight$"
    )
    return build_bundle(
        groups,
        torch.device("cpu"),
        "adamw",
        {"lr": 1e-2, "betas": (0.9, 0.95), "eps": 1e-8},
        SCHED,
        10,
        build_policy(precision, ()),
        0.0,
        fused_step=fused,
    )


def _train(fused: bool, steps: int = 3, x=None):
    model = _model()
    bundle = _bundle(model, fused)
    x = (
        torch.randn(4, 6, generator=torch.Generator().manual_seed(1))
        if x is None
        else x
    )
    for _ in range(steps):
        bundle.backward(model(x).square().mean())
        if fused:
            for p in model.parameters():
                bundle.step_param(p)
        bundle.step()
    return model, bundle


def _same(a, b) -> None:
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_fused_matches_plain_with_decay_to_one():
    plain, pb = _train(False)
    fused, fb = _train(True)
    assert float(plain[1].weight.mean()) < 3.0 - 1e-3  # decayed towards one
    _same(plain, fused)
    no_decay = _model()
    must_fail(lambda: _same(no_decay, fused), "parameters not stepped")


def test_fused_reports_the_gradient_norm():
    plain, pb = _train(False, steps=1)
    fused, fb = _train(True, steps=1)
    assert fb.last_norm == pytest.approx(pb.last_norm, rel=1e-5)
    must_fail(lambda: _close(float("nan"), pb.last_norm), "norm reported as NaN")


def _close(a: float, b: float) -> None:
    assert a == pytest.approx(b, rel=1e-5)


def test_nonfinite_fused_step_is_applied_and_counted():
    model = _model()
    bundle = _bundle(model, True, "fp16")
    before = [p.detach().clone() for p in model.parameters()]
    scale0 = bundle.scaler.scale
    bundle.backward(model(torch.randn(4, 6)).square().mean())
    first = list(model.parameters())[0]
    first.grad.view(-1)[0] = float("inf")
    for p in model.parameters():
        bundle.step_param(p)
    assert bundle.step() is True
    assert bundle.skipped == 0 and bundle.nonfinite == 1 and not bundle.last_ok
    assert bundle.scaler.scale == scale0 / 2
    moved = [not torch.equal(p, q) for p, q in zip(model.parameters(), before)]
    assert all(moved[1:])


def test_plain_bundle_counts_nonfinite_as_skipped():
    model = _model()
    bundle = _bundle(model, False, "fp16")
    bundle.backward(model(torch.full((2, 6), float("inf"))).sum())
    assert bundle.step() is False
    assert bundle.skipped == 1 and bundle.nonfinite == 1
