"""Every knob a bench run sets is a knob of its script (the scripts reject unknown
knobs, so a renamed knob would fail the row at start), for every run kind and pool."""

from pathlib import Path

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.plan import matrix
from kohakumoss_bench.plan.config import Card, Host, Options, Run
from kohakumoss_bench.plan.coverage_runs import coverage_runs
from kohakumoss_bench.plan.knobs import SAMPLE, script_defaults
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
REPO = Path(__file__).resolve().parents[3]


def host() -> Host:
    return Host(
        name="h",
        root="/r",
        python="python",
        weights={"source": "/w/base", "stage": "/s/base"},
        fix_file={"source": "/w/fix.safetensors", "stage": "/s/fix.safetensors"},
        datasets=[{"source": "kohya", "path": "/d"}],
        cache_dir="/c",
        ncu=["ncu"],
    )


def check_known(runs: list[Run]) -> None:
    known: dict[str, set] = {}
    for run in runs:
        if run.script not in known:
            known[run.script] = set(script_defaults(REPO / run.script))
        unknown = sorted(set(run.knobs) - known[run.script])
        assert not unknown, f"{run.name} ({run.script}): unknown knobs {unknown}"


def _queues() -> list[Run]:
    out = []
    for card in (Card(0, "5090", 32.0, 120), Card(0, "2080ti", 22.0, 75)):
        cov = coverage_runs(host(), Options(), card)
        out += matrix.card_queue(host(), Options(), card, cov)
    return out


def test_every_run_knob_is_a_script_knob():
    runs = _queues()
    assert {r.kind for r in runs} >= {"speed", "quality", "export", "grid", "infer"}
    check_known(runs)


def test_negative_an_old_knob_name():
    broken = Run("infer-old", "infer", SAMPLE, {"RESIDENCY": "offload"}, 1)
    must_fail(lambda: check_known([broken]), "RESIDENCY on sample.py")
