"""The text-encoder memory model: per-placement footprints and the `auto` choice."""

import pytest

from kohakumoss.offload.encoders import text_memory
from kohakumoss.offload.encoders.text_memory import (
    TextSizes,
    choose_text_memory,
    text_footprint,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

GIB = 1 << 30
SIZES = TextSizes(
    total=16 * GIB, embedding=2 * GIB, layers=13 * GIB, ring=GIB, hidden=4096,
    intermediate=12288, value_heads=32, head_k=128, head_v=128, elem=2,
)  # fmt: skip
TOKENS = (16384, 16384)


def _pick(budget: int, step: int) -> str:
    return choose_text_memory(SIZES, budget, step, *TOKENS)[0]


def test_footprints_order_by_what_stays_on_the_device():
    fp = {
        m: text_footprint(m, SIZES, *TOKENS) for m in ("resident", "offload", "stream")
    }
    assert fp["resident"]["resident"] == SIZES.total
    assert fp["offload"]["resident"] == 0
    assert fp["stream"]["resident"] == SIZES.total - SIZES.embedding - SIZES.layers
    assert fp["stream"]["pass"] < fp["offload"]["pass"] < fp["resident"]["pass"]


def test_auto_picks_resident_when_it_fits_next_to_the_step_else_stream(monkeypatch):
    assert _pick(40 * GIB, 8 * GIB) == "resident"
    assert _pick(20 * GIB, 8 * GIB) == "stream"
    assert _pick(8 * GIB, 6 * GIB) == "stream"

    def never_offload():
        for budget in (8, 12, 16, 20, 24, 32, 48):
            assert _pick(budget * GIB, 8 * GIB) != "offload", budget

    never_offload()
    monkeypatch.setattr(text_memory, "ORDER", ("resident", "offload", "stream"))
    must_fail(never_offload, "offload ranked before stream")
