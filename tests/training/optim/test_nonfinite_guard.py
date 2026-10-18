"""Non-finite gradients are never silent: the fused-step bundle reports a NaN gradient
norm and the count of non-finite parameters when any gradient was non-finite (the
non-fused bundle's behaviour), the metric row carries `nonfinite_params`, and
NonFiniteGuard stops a run whose gradients stay non-finite. Model-free: plain
parameters and hand-built step states."""

import math

import pytest
import torch
from torch import nn

from kohakumoss.precision.scaler import NoScaler
from kohakumoss.training.callbacks.finite import NonFiniteGuard
from kohakumoss.training.callbacks.monitor import LogMetrics
from kohakumoss.training.drivers.common import record
from kohakumoss.training.optim.bundle.fused import FusedStepBundle
from kohakumoss.training.optim.bundle.nonfinite import nonfinite_count
from kohakumoss.training.state import TrainState
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
PATIENCE = 8


def _bundle(params: list[nn.Parameter]) -> FusedStepBundle:
    groups = [{"params": params, "lr": 1e-3, "names": ["a", "b", "c"]}]
    return FusedStepBundle(
        lambda g: torch.optim.SGD(g, lr=1e-3),
        lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
        NoScaler(),
        groups,
        device=torch.device("cpu"),
    )


def _fused_step(bad: bool) -> tuple[FusedStepBundle, float]:
    """One in-backward step over three parameters, the second one NaN when `bad`;
    returns the bundle and the norm of the finite gradients alone."""
    torch.manual_seed(0)
    params = [nn.Parameter(torch.randn(4, 4)) for _ in range(3)]
    grads = [torch.randn(4, 4) for _ in params]
    if bad:
        grads[1][0, 0] = float("nan")
    bundle = _bundle(params)
    for p, g in zip(params, grads):
        p.grad = g.clone()
        bundle.step_param(p)
    bundle.step()
    finite = math.sqrt(
        sum(float(g.square().sum()) for g in grads if g.isfinite().all())
    )
    return bundle, finite


def _reports_nonfinite(norm: float, count: int) -> None:
    assert math.isnan(norm), norm
    assert count == 1, count


def test_fused_bundle_reports_nan_norm_and_count():
    bundle, finite = _fused_step(bad=True)
    assert not bundle.last_ok
    _reports_nonfinite(bundle.last_norm, bundle.last_nonfinite_params)
    assert [g["param"] for g in bundle.nonfinite_grads] == ["b"]
    must_fail(
        lambda: _reports_nonfinite(finite, 0), "the norm of the finite gradients alone"
    )
    clean, finite = _fused_step(bad=False)
    assert clean.last_ok and clean.last_nonfinite_params == 0
    assert clean.last_norm == pytest.approx(finite, rel=1e-5)


class _Rows:
    def __init__(self):
        self.rows = []

    def log(self, metrics: dict, step: int) -> None:
        self.rows.append(metrics)


class _Out:
    metrics: dict = {}


def test_metric_row_carries_nonfinite_params():
    bundle, _ = _fused_step(bad=True)
    state = TrainState(step=1)
    record(state, bundle)
    assert state.nonfinite_params == 1 and state.nonfinite_names == ["b"]
    sink = _Rows()
    LogMetrics(1, sink).on_step_end(state, _Out())
    assert sink.rows[0]["nonfinite_params"] == 1.0
    assert math.isnan(sink.rows[0]["grad_norm"])


def test_nonfinite_count():
    good, bad = torch.ones(3), torch.tensor([1.0, float("inf"), 0.0])
    assert nonfinite_count([good, bad, bad]) == 2
    assert nonfinite_count([]) == 0


def _states(n: int, scale, names) -> list[TrainState]:
    return [
        TrainState(
            step=i + 1,
            ok=False,
            scale=scale(i),
            nonfinite_params=len(names(i)),
            nonfinite_names=list(names(i)),
        )
        for i in range(n)
    ]


def _stops(guard: NonFiniteGuard, states: list[TrainState]) -> int:
    """The step the guard stopped at (AssertionError when it never stops)."""
    for st in states:
        try:
            guard.on_step_end(st, _Out())
        except FloatingPointError:
            return st.step
    raise AssertionError("the guard never stopped the run")


def test_guard_stops_at_the_scale_floor():
    varying = _states(20, lambda i: 1.0, lambda i: [f"p{i}"])
    assert _stops(NonFiniteGuard(PATIENCE, 1.0), varying) == PATIENCE
    must_fail(
        lambda: _stops(NonFiniteGuard(10**9, 1.0), varying), "a guard that never fires"
    )


def test_guard_stops_on_repeating_parameters_above_the_floor():
    halving = _states(20, lambda i: 2.0 ** (16 - i), lambda i: ["blocks.3.attn.q"])
    assert _stops(NonFiniteGuard(PATIENCE, 1.0), halving) == PATIENCE
    moving = _states(PATIENCE + 2, lambda i: 2.0 ** (16 - i), lambda i: [f"p{i}"])
    must_fail(
        lambda: _stops(NonFiniteGuard(PATIENCE, 1.0), moving),
        "different parameters each step, scale still backing off",
    )


def test_guard_resets_on_a_finite_step():
    guard = NonFiniteGuard(PATIENCE, 1.0)
    bad = _states(PATIENCE - 1, lambda i: 1.0, lambda i: ["x"])
    for st in bad:
        guard.on_step_end(st, _Out())
    guard.on_step_end(TrainState(step=PATIENCE), _Out())
    again = _states(PATIENCE - 1, lambda i: 1.0, lambda i: ["x"])
    must_fail(lambda: _stops(guard, again), "a guard counting across a finite step")
