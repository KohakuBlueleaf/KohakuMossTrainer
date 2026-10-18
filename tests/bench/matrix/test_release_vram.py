"""Release rows report the prewarm phase against the step phase (nvidia-smi peak, torch
max allocated / reserved; step 1's torch peak from its log line, every later step's
from its metric row) and flag a row whose prewarm peaked above its steps: tuning must
never use more VRAM than training."""

import json
import statistics

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.report import results
from kohakumoss_bench.report.build import release_rows
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
LOG = (
    "[18:40:01] [training.build.kernels] [INFO] VRAM peak: prewarm 3.00 GiB allocated / "
    "3.50 reserved; step 1 6.00 / 7.00\n"
)


def _record(tmp_path, prewarm_smi: float, steps: list[float], prewarm_torch=3.0):
    """A finished release row: its log, its metric rows (step 1, then `steps` as the
    per-step allocated peaks, reserved 0.5 above), its nvidia-smi phases."""
    (tmp_path / "train").mkdir(parents=True)
    rows = [
        {"step": 1, "prewarm_peak_alloc": prewarm_torch, "prewarm_peak_reserved": 3.5}
    ]
    rows += [
        {
            "step": i,
            "vram_step_peak": v,
            "vram_peak_alloc": v,
            "vram_peak_reserved": v + 0.5,
        }
        for i, v in enumerate(steps, start=2)
    ]
    (tmp_path / "train" / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    log = tmp_path / "log.txt"
    log.write_text(LOG, encoding="utf-8")
    summary = {**results.log_summary(log), **results.training_summary(tmp_path, 0)}
    gib = 1024.0
    smi = {
        "prewarm_smi_process_peak_mib": prewarm_smi * gib,
        "prewarm_smi_process_steady_mib": 2.0 * gib,
        "step_smi_process_peak_mib": (max(steps) + 1.5) * gib,
        "step_smi_process_steady_mib": (statistics.median(steps) + 1.0) * gib,
    }
    return {
        "kind": "release",
        "run": "release-x",
        "status": "ok",
        "summary": summary,
        "vram_smi_phases": smi,
    }


def test_step_one_line_parsed(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(LOG, encoding="utf-8")
    s = results.log_summary(log)
    assert (s["step1_peak_alloc"], s["step1_peak_reserved"]) == (6.0, 7.0)


def check_flag(flag, tmp_path) -> None:
    spike = _record(tmp_path / "spike", prewarm_smi=12.0, steps=[5.0, 5.0, 5.0])
    quiet = _record(tmp_path / "quiet", prewarm_smi=4.0, steps=[5.0, 5.0, 5.0])
    assert flag(spike) is True, "a prewarm spike above the step peak is not flagged"
    assert flag(quiet) is False, "a prewarm below the steps is flagged"


def check_long_pack(flag, tmp_path) -> None:
    """One long pack's step (11 GiB) is above the torch prewarm (9 GiB), the other
    steps below it: the steps peaked higher, so the row is not flagged."""
    row = _record(tmp_path / "long", 4.0, [5.0, 5.0, 5.0, 11.0], prewarm_torch=9.0)
    assert flag(row) is False, "a row whose longest step tops the prewarm is flagged"


def _over(record):
    return results.phase_peaks(record)["over"]


def test_prewarm_over_step_flag(tmp_path):
    check_flag(_over, tmp_path)
    check_long_pack(_over, tmp_path / "max")
    line = release_rows([_record(tmp_path / "line", 12.0, [5.0, 5.0])])[-1]
    assert "| yes |" in line and "| 3.00 / 3.50 | 6.00 / 7.00 |" in line, line
    assert results.phase_peaks({"summary": {}})["over"] is None


def test_negative_steady_only_flag(tmp_path):
    def steady_only(record):
        smi = record["vram_smi_phases"]
        return (
            smi["prewarm_smi_process_steady_mib"] > smi["step_smi_process_steady_mib"]
        )

    must_fail(lambda: check_flag(steady_only, tmp_path), "flag from steady values only")


def check_budget(flag, tmp_path) -> None:
    """A vram_8g row whose reserved grows 6.5 -> 8.5 GiB over the steps went past its
    8 GB cap; one that stays at 3.5 (step 1 at 7.0) did not."""
    grows = _record(tmp_path / "grows", 4.0, [6.0, 7.0, 8.0])
    flat = _record(tmp_path / "flat", 4.0, [3.0, 3.0, 3.0])
    for row in (grows, flat):
        row["cap_gib"] = 8.0
        row["vram_smi_phases"]["step_smi_process_peak_mib"] = 5.0 * 1024
    assert flag(grows) is True and flag(flat) is False, (flag(grows), flag(flat))


def test_reserved_growth_and_budget(tmp_path):
    check_budget(lambda r: results.phase_peaks(r)["past_budget"], tmp_path / "b")
    p = results.phase_peaks(_record(tmp_path / "trend", 4.0, [4.0, 5.0, 6.0]))
    assert (p["reserved_first"], p["reserved_last"]) == (4.5, 6.5)

    def first_step_only(r):
        return results.phase_peaks(r)["reserved_first"] > r["cap_gib"]

    must_fail(lambda: check_budget(first_step_only, tmp_path / "n"), "first step only")


def test_negative_median_step_peak(tmp_path):
    def median_steps(record):
        s = record["summary"]
        return s["prewarm_peak_alloc_max"] > s["vram_step_peak_median"]

    must_fail(lambda: check_long_pack(median_steps, tmp_path), "median of step peaks")
