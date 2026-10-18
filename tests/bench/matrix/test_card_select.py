"""Bench cards are chosen by PCI bus: when a card drops off the bus the CUDA indices
shift, so a fixed index list picks the wrong cards; a listed bus without a card refuses
the start."""

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.measure import smi
from kohakumoss_bench.plan.config import Card
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
BUSES = ["83", "84", "C3", "C4"]


def _found() -> list[Card]:
    """.77 after a 32 GB card left the bus: three 32 GB, then the four 16 GB."""
    big = [
        Card(i, "PG503", 32.0, 70, bus=f"00000000:{b}:00.0")
        for i, b in enumerate(("44", "45", "46"))
    ]
    small = [
        Card(3 + i, "V100 16GB", 16.0, 70, bus=f"00000000:{b}:00.0")
        for i, b in enumerate(BUSES)
    ]
    return big + small


def check_small_pool(chosen: list[Card]) -> None:
    assert [c.total_gib for c in chosen] == [16.0] * 4, [c.index for c in chosen]
    assert [smi.bus_number(c.bus) for c in chosen] == [b.lower() for b in BUSES]


def test_cards_by_bus_after_an_index_shift():
    check_small_pool(smi.select(_found(), None, BUSES))
    check_small_pool(smi.select(_found(), [3, 4, 5, 6], BUSES))
    assert smi.bus_number("0000:C3:00.0") == "c3" and smi.bus_number("C3") == "c3"


def test_refuses_a_missing_bus_or_a_stale_index_list():
    with pytest.raises(ValueError, match="no card on PCI bus"):
        smi.select(_found(), None, [*BUSES, "c5"])
    with pytest.raises(ValueError, match="not gpus"):
        smi.select(_found(), [4, 5, 6, 7], BUSES)


def test_negative_the_old_index_list():
    stale = smi.select(_found(), [4, 5, 6, 7], None)
    must_fail(lambda: check_small_pool(stale), "indices 4-7 after the shift")
