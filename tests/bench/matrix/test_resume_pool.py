"""Resume reads only the pool's own cards (pools of two card models share run names),
and the encoder columns are means over the steady steps (an encoder running `ahead`
batches at once costs 0 on most steps, so their median is 0)."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.plan.config import Options
from kohakumoss_bench.report import results
from kohakumoss_bench.report.build import cache_rows
from kohakumoss_bench.run.host import FINAL, HostRunner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _record(root, card: str, name: str) -> None:
    d = root / card / name
    d.mkdir(parents=True)
    record = {"run": name, "status": "ok", "dir": str(d), "card": card}
    (d / "result.json").write_text(json.dumps(record), encoding="utf-8")


def _any_card(self, name, model):
    """Broken: a record from any card."""
    for path in sorted(self.out.glob(f"*/{name}/result.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") in FINAL:
            return record
    return None


def check_own_pool(finished, tmp_path) -> None:
    _record(tmp_path, "2-NVIDIA_GeForce_RTX_2080_Ti", "speed-fp16-x")
    _record(tmp_path, "0-NVIDIA_GeForce_RTX_3090", "speed-fp16-y")
    fake = SimpleNamespace(opts=Options(resume=True), out=tmp_path)
    assert (
        finished(fake, "speed-fp16-x", "NVIDIA_GeForce_RTX_3090") is None
    ), "a 3090 row resumed from a 2080 Ti result"
    own = finished(fake, "speed-fp16-y", "NVIDIA_GeForce_RTX_3090")
    assert own is not None and own["card"].endswith("RTX_3090")


def test_resume_reads_the_pools_own_cards(tmp_path):
    check_own_pool(HostRunner.finished, tmp_path)


def test_negative_resume_from_any_card(tmp_path):
    must_fail(lambda: check_own_pool(_any_card, tmp_path), "resume from any card")


def _summary(tmp_path) -> dict:
    rows = [
        {"step": i, "te_time": 0.8 if i % 8 == 0 else 0.0, "te_share": 0.0}
        for i in range(1, 51)
    ]
    path = tmp_path / "train" / "metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return results.training_summary(tmp_path, 10)


def check_encoder_column(line: str) -> None:
    assert "| 0.100 (0.00) |" in line, f"the te column is not the mean: {line}"


def test_cache_table_shows_encoder_means(tmp_path):
    s = _summary(tmp_path)
    assert s["te_time_median"] == 0.0 and s["te_time_mean"] == pytest.approx(0.1)
    record = {"kind": "cache", "run": "r", "tags": {}, "summary": s}
    check_encoder_column(cache_rows([record])[-1])
    medians = {**s, "te_time_mean": s["te_time_median"]}
    broken = cache_rows([{**record, "summary": medians}])[-1]
    must_fail(lambda: check_encoder_column(broken), "median encoder column")
