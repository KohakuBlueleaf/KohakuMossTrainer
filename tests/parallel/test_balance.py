"""Token budgets and stage splits for heterogeneous devices."""

import itertools

import pytest

from kohakumoss.parallel.balance import split_layers, token_budgets
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_token_budgets_proportional():
    out = token_budgets(16384, [71.0, 35.5])
    assert sum(out) == 16384
    assert abs(out[0] / out[1] - 2.0) < 0.1
    assert all(v % 256 == 0 for v in out)


def _brute(costs, speeds):
    n, k = len(costs), len(speeds)
    best = None
    for cuts in itertools.combinations(range(1, n), k - 1):
        edges = (0, *cuts, n)
        val = max(
            sum(costs[a:b]) / s for (a, b), s in zip(zip(edges, edges[1:]), speeds)
        )
        best = val if best is None else min(best, val)
    return best


@pytest.mark.parametrize("speeds", [[1.0, 1.0], [2.0, 1.0], [1.0, 3.0, 2.0]])
def test_split_layers_optimal(speeds):
    costs = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0]
    bounds = split_layers(costs, speeds)
    got = max(sum(costs[a:b]) / s for (a, b), s in zip(bounds, speeds))
    assert abs(got - _brute(costs, speeds)) < 1e-12
    assert bounds[0][0] == 0 and bounds[-1][1] == len(costs)


def test_even_split_is_caught_as_suboptimal():
    costs, speeds = [1.0] * 6, [2.0, 1.0]
    assert split_layers(costs, speeds) == [(0, 4), (4, 6)]

    def check_even_split():
        even = max(3 / speeds[0], 3 / speeds[1])
        assert abs(even - _brute(costs, speeds)) < 1e-12

    must_fail(check_even_split, "an even split is not optimal for 2:1 speeds")


def test_split_respects_caps():
    bounds = split_layers([1.0] * 6, [10.0, 1.0], caps=[3.0, 6.0])
    assert bounds[0][1] <= 3
