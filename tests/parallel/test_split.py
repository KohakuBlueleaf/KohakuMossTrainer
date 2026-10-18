"""Pipeline stage split (pure functions): the DP partition is the brute-force optimum
with the prologue / epilogue costs on the end stages, the V placement balances each
rank's pair of stages, and explicit layer counts become ranges."""

import itertools

import pytest

from kohakumoss.parallel.pipeline.split import (
    StageCosts,
    bounds_from_counts,
    partition,
    partition_v,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
LAYERS = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0]


def _load(costs: StageCosts, bounds, speeds) -> float:
    n = len(bounds)
    loads = []
    for s, ((lo, hi), v) in enumerate(zip(bounds, speeds)):
        extra = (costs.first if s == 0 else 0.0) + (costs.last if s == n - 1 else 0.0)
        loads.append((sum(costs.layers[lo:hi]) + extra) / v)
    return max(loads)


def _brute(costs: StageCosts, speeds) -> float:
    n, k = len(costs.layers), len(speeds)
    best = float("inf")
    for cuts in itertools.combinations(range(1, n), k - 1):
        edges = (0, *cuts, n)
        best = min(best, _load(costs, list(zip(edges, edges[1:])), speeds))
    return best


def check_optimal(costs: StageCosts, bounds, speeds) -> None:
    assert bounds[0][0] == 0 and bounds[-1][1] == len(costs.layers), bounds
    assert all(lo < hi for lo, hi in bounds), bounds
    assert abs(_load(costs, bounds, speeds) - _brute(costs, speeds)) < 1e-12


@pytest.mark.parametrize("speeds", [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
@pytest.mark.parametrize("ends", [(0.0, 0.0), (3.0, 0.0), (1.0, 4.0)])
def test_partition_is_optimal(speeds, ends):
    costs = StageCosts(LAYERS, ends[0], ends[1], [1.0] * len(LAYERS))
    check_optimal(costs, partition(costs, len(speeds), speeds), speeds)


def test_prologue_takes_layers_off_the_first_stage():
    costs = StageCosts([1.0] * 6, 2.0, 0.0, [1.0] * 6)
    assert partition(costs, 2) == [(0, 2), (2, 6)]
    blind = partition(StageCosts([1.0] * 6, 0.0, 0.0, [1.0] * 6), 2)
    must_fail(lambda: check_optimal(costs, blind, [1.0, 1.0]), "prologue ignored")


def test_ties_break_on_parameters():
    costs = StageCosts([1.0] * 4, 0.0, 0.0, [1.0, 2.0, 3.0, 9.0])
    assert partition(costs, 3) == [(0, 2), (2, 3), (3, 4)]


def test_v_placement_balances_rank_pairs():
    costs = StageCosts([1.0] * 8, 1.0, 1.0, [1.0] * 8)
    bounds = partition_v(costs, 2)
    assert bounds == [(0, 2), (2, 5), (5, 7), (7, 8)]
    rank0 = (bounds[0][1] - bounds[0][0]) + (bounds[3][1] - bounds[3][0]) + 2.0
    rank1 = (bounds[1][1] - bounds[1][0]) + (bounds[2][1] - bounds[2][0])
    assert rank0 == rank1 == 5.0


def test_counts_become_ranges():
    assert bounds_from_counts([3, 2, 4]) == [(0, 3), (3, 5), (5, 9)]
