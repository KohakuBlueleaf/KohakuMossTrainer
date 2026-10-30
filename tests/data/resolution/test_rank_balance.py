"""DATA_BALANCE: each step's `world x GRAD_ACC` batches are dealt to ranks longest-cost
first, every batch once, each rank GRAD_ACC of them, and the most loaded rank carries
less than under round robin; off (or one rank) it is the round robin. Model-free:
a planner with hand-set batch costs."""

import pytest

from kohakumoss_data.resolution.base import Planner
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]
COSTS = [9.0, 1.0, 1.0, 1.0, 8.0, 1.0, 2.0, 1.0, 7.0, 3.0, 1.0, 1.0, 5.0, 4.0, 1.0, 1.0]


class _Costed(Planner):
    """Planner shell: global batch g costs COSTS[g] (no catalog)."""

    def __init__(self, balance: int):
        self.balance = balance
        self.attn_tokens = 1
        self._blocks = {}

    def cost(self, g: int) -> float:
        return COSTS[g]


def _loads(planner, world: int, per: int) -> list[float]:
    ids = [
        [planner.global_batch(i, r, world) for i in range(per)] for r in range(world)
    ]
    flat = sorted(g for rank in ids for g in rank)
    assert flat == list(range(world * per)), flat
    assert all(len(rank) == per for rank in ids)
    return [sum(COSTS[g] for g in rank) for rank in ids]


def check_balanced(planner, world: int = 4, per: int = 4) -> None:
    balanced = max(_loads(planner, world, per))
    round_robin = max(_loads(_Costed(0), world, per))
    assert balanced < round_robin, (balanced, round_robin)


def test_balanced_block_beats_round_robin():
    check_balanced(_Costed(4))
    assert max(_loads(_Costed(4), 4, 4)) == pytest.approx(sum(COSTS) / 4, abs=3.0)


def test_negative_balance_off():
    must_fail(lambda: check_balanced(_Costed(0)), "round robin")


def test_one_rank_is_round_robin():
    p = _Costed(4)
    assert [p.global_batch(i, 0, 1) for i in range(6)] == list(range(6))
