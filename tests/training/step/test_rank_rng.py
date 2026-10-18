"""Per-rank streams: each data-parallel rank draws its own sigma / noise / data stream,
every pipeline stage draws rank 0's."""

import dataclasses

import pytest

from kohakumoss.training.layouts import LAYOUTS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_streams(layouts) -> None:
    assert [layouts["ddp"].stream_rank(r) for r in (0, 1)] == [0, 1]
    assert [layouts["fsdp"].stream_rank(r) for r in (0, 1)] == [0, 1]
    assert [layouts["pipeline"].stream_rank(r) for r in (0, 1)] == [0, 0]
    assert layouts["threaded"].stream_rank(0) == 0 == layouts["none"].stream_rank(0)
    assert [layouts[n].data_world(4) for n in ("ddp", "fsdp", "pipeline")] == [4, 4, 1]


def test_stream_rank_follows_data_rank():
    layouts = {name: LAYOUTS.get(name) for name in LAYOUTS.names()}
    check_streams(layouts)
    shared = dataclasses.replace(layouts["pipeline"], shared_data=False)
    broken = {**layouts, "pipeline": shared}
    must_fail(lambda: check_streams(broken), "pipeline ranks drawing their own data")
