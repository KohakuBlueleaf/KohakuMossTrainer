"""The precision bar itself: both-metrics-worse rule, non-finite, nonzero-ref relative."""

import pytest
import torch

from kohakumoss_kernels.precision import FACTOR, error, judge
from tests._util.negative import must_fail
from tests._util.precision import assert_bar

pytestmark = pytest.mark.cpu

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def ref64(n: int = 4096) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    r = torch.randn(n, generator=g, dtype=torch.float64)
    return r * torch.logspace(-3, 3, n, dtype=torch.float64)


def scaled(eager: torch.Tensor, r: torch.Tensor, k: float) -> torch.Tensor:
    """A result whose error is `k` x the eager error on every element (in fp64)."""
    return r + k * (eager.double() - r)


@pytest.mark.parametrize("dtype", DTYPES)
def test_equal_error_passes(dtype):
    r = ref64()
    e = r.to(dtype)
    assert judge(e, e, r).ok
    assert_bar(e, e, r)


@pytest.mark.parametrize("dtype", DTYPES)
def test_both_metrics_worse_fails(dtype):
    r = ref64()
    e = r.to(dtype)
    worse = scaled(e, r, 2 * FACTOR)
    assert not judge(worse, e, r).ok
    must_fail(lambda: assert_bar(worse, e, r), "both metrics 4x eager")


def test_one_metric_worse_passes():
    """Worse absolute error on a large element only (relative error unchanged)."""
    r = torch.tensor([1e-3, 1.0, 1e3], dtype=torch.float64)
    eager = r + torch.tensor([1e-6, 1e-4, 1e-2], dtype=torch.float64)
    out = r + torch.tensor([1e-6, 1e-4, 1e-1], dtype=torch.float64)
    v = judge(out, eager, r)
    assert v.out.abs > FACTOR * v.eager.abs and v.out.rel <= FACTOR * v.eager.rel
    assert v.ok


def test_non_finite_fails():
    r = ref64()
    out = r.float()
    out[3] = float("nan")
    assert not judge(out, r.float(), r).ok


def test_relative_skips_zero_reference():
    r = torch.tensor([0.0, 2.0], dtype=torch.float64)
    out = torch.tensor([1e-3, 2.0], dtype=torch.float64)
    err = error(out, r)
    assert err.abs == pytest.approx(1e-3) and err.rel == 0.0
