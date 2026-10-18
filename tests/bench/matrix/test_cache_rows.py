"""The flagship cache-mode rows (head of rank 1 on every pool, fp16 equivalents only on
cards without bf16) and the nvidia-smi split into the prewarm phase and the step loop.
"""

import json

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.measure.smi import Trace
from kohakumoss_bench.plan import matrix
from kohakumoss_bench.plan.config import Card, Host, Options
from kohakumoss_bench.report import results
from kohakumoss_bench.report.build import cache_rows
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CARDS = {
    "5090": (Card(0, "5090", 32.0, 120), 6),
    "3090": (Card(0, "3090", 24.0, 86), 5),
    "2080ti": (Card(0, "2080ti", 22.0, 75), 4),
    "v100": (Card(0, "v100", 16.0, 70), 4),
}


def host() -> Host:
    return Host(
        name="h",
        root="/r",
        python="python",
        weights={"source": "/w/base", "stage": "/s/base"},
        datasets=[{"source": "kohya", "path": "/d"}],
        cache_dir="/c",
    )


def check_cache_rows(queue: list, card: Card, setups: int) -> None:
    rows = [r for r in queue if r.kind == "cache"]
    assert len(rows) == 3 * setups, (card.name, len(rows))
    assert queue[1 : 1 + len(rows)] == rows, "cache rows not at the head of rank 1"
    if not card.bf16:
        assert {r.tags["precision"] for r in rows} == {"fp16"}
    for r in rows:
        mode = matrix.CACHE_MODES[r.tags["cache"]]
        assert {k: r.knobs[k] for k in mode} == mode, r.name
    user = [r for r in rows if r.tags["profile"] == matrix.LOWVRAM_LOKR_PROFILE]
    assert len(user) == 3 and all(
        r.knobs["TEXT_ENCODE"] == matrix.LOWVRAM_LOKR_TEXT for r in user
    )
    assert all(
        (r.tags["precision"], r.tags["base"], r.tags["algo"])
        == ("fp16", "int8", "lokr16")
        for r in user
    )
    fp8 = [r for r in rows if r.tags["base"] == "fp8_compute"]
    assert len(fp8) == (3 if card.fp8 else 0), card.name


def test_cache_rows_head_rank_one_on_every_pool():
    for card, setups in CARDS.values():
        queue = matrix.card_queue(host(), Options(), card, [])
        check_cache_rows(queue, card, setups)


def test_cache_row_check_catches_a_late_or_partial_queue():
    card, setups = CARDS["5090"]
    queue = matrix.card_queue(host(), Options(), card, [])
    batch = [r for r in queue if r.kind == "batch"]
    late = [queue[0], *batch, *[r for r in queue[1:] if r.kind != "batch"]]
    must_fail(lambda: check_cache_rows(late, card, setups), "cache rows after batch")
    partial = [r for r in queue if r.tags.get("cache") != "nocache"]
    must_fail(lambda: check_cache_rows(partial, card, setups), "a cache mode missing")


def _trace() -> Trace:
    rows = [(0.5, 2000.0, 2500.0), (1.5, 20000.0, 20500.0), (2.5, 9000.0, 9500.0)]
    rows += [(3.5 + i, 8000.0, 8500.0) for i in range(6)]
    return Trace("gpu", 1, 500.0, 0.0, rows)


def check_split(phases: dict) -> None:
    assert phases["prewarm_smi_process_peak_mib"] == 20000.0, phases
    assert phases["step_smi_process_peak_mib"] == 9000.0, phases
    assert phases["step_smi_process_steady_mib"] == 8000.0, phases
    assert phases["step_smi_card_above_idle_mib"] == 9000.0, phases


def test_smi_split_at_the_loop_start(tmp_path):
    path = tmp_path / "train" / "metrics.jsonl"
    path.parent.mkdir(parents=True)
    rows = [{"step": 1, "time": 1000.0, "step_time": 1.5}, {"step": 2, "time": 1001.0}]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    start = results.loop_start(tmp_path)
    assert start == 998.5
    started = 996.0
    phases = _trace().phases(start - started)
    check_split(phases)
    record = {"kind": "cache", "run": "r", "tags": {}, "summary": {}}
    record["vram_smi_phases"] = phases
    line = cache_rows([record])[-1]
    assert "| 19.53 / " in line and "| 8.79 / 7.81 |" in line, line
    must_fail(
        lambda: check_split(_trace().phases(0.0)), "no split: prewarm in the loop"
    )
    assert _trace().phases(None) == {}
