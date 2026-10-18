"""Real-weights memory profiles: over a short LoRA run on a slice of the
KOHAKUMOSS_DATASET dataset (bucketed sizes, so the token count varies), the allocator's
reserved memory stays within a bound of the allocated peak and does not grow on a step
that brings no larger batch than every earlier one; the profiles order by this
process's device use (what nvidia-smi shows): minimal < low < balanced < speed, for
every profile that fits the card; and a finished
run leaves no host worker thread behind. Each run is its own process (a fresh allocator,
device baseline and thread set)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._util.dataset import kohya_slice
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[3]
CHILD = "tests.memory.real._profile_child"
ORDER = ("minimal", "low", "balanced", "speed")
SLICE = 64
SLACK = 1.15  # reserved / allocated peak
PAGES = 0.25  # GiB of allocator rounding on top
GROWTH = 0.25  # GiB reserved may grow on steps that bring no larger batch


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    root = os.environ.get("KOHAKUMOSS_DATASET")
    if not root:
        pytest.skip("KOHAKUMOSS_DATASET unset")
    return kohya_slice(Path(root), tmp_path_factory.mktemp("dataset"), SLICE)


def _run(tmp: Path, dataset: Path, profile: str, variant: str = ""):
    """(metric rows, worker threads left) of one run, or None when it does not fit."""
    argv = [sys.executable, "-m", CHILD, profile, str(tmp), str(dataset)]
    out = subprocess.run(
        argv + ([variant] if variant else []),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    if result.get("oom"):
        return None
    lines = Path(result["rows"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()], result["threads"]


def _tokens(row: dict) -> float:
    return row["tokens_per_s"] * row["step_time"]


def creep(rows: list[dict]) -> float:
    """GiB the reserved peak grew, from step 2 on, on steps whose batch is no larger
    than every earlier batch (growth on a new largest batch is the allocator meeting a
    new size, not a creep)."""
    grown, largest = 0.0, _tokens(rows[0])
    for prev, row in zip(rows, rows[1:]):
        step_up = row["vram_peak_reserved"] - prev["vram_peak_reserved"]
        if _tokens(row) <= largest:
            grown += max(step_up, 0.0)
        largest = max(largest, _tokens(row))
    return grown


def _no_creep(rows: list[dict]) -> None:
    reserved = [(round(_tokens(r)), r["vram_peak_reserved"]) for r in rows]
    assert creep(rows) <= GROWTH, (creep(rows), reserved)


def _bounded(rows: list[dict]) -> None:
    peak = max(r["vram_peak_alloc"] for r in rows)
    held = max(r["vram_peak_reserved"] for r in rows)
    assert held <= SLACK * peak + PAGES, (held, peak)
    _no_creep(rows)


def _with_creep(rows: list[dict], gib: float) -> list[dict]:
    """`rows` with the reserved peak `gib` higher from the last step that brings no
    larger batch on (a creep with no larger batch behind it)."""
    flat = [
        i
        for i in range(1, len(rows))
        if _tokens(rows[i]) <= max(_tokens(r) for r in rows[:i])
    ]
    assert flat, "every step brought a larger batch: no creep can be placed"
    return [
        {
            **r,
            "vram_peak_reserved": r["vram_peak_reserved"]
            + (gib if i >= flat[-1] else 0),
        }
        for i, r in enumerate(rows)
    ]


def _ordered(device: dict[str, float]) -> None:
    fitted = [p for p in ORDER if p in device]
    uses = [device[p] for p in fitted]
    assert uses == sorted(uses) and len(set(uses)) == len(uses), dict(device)


def _no_threads(left: int) -> None:
    assert left == 0, f"{left} host worker threads left after the run"


@pytest.fixture(scope="module")
def runs(tmp_path_factory, dataset):
    tmp = tmp_path_factory.mktemp("profiles")
    done = {p: _run(tmp, dataset, p) for p in ORDER}
    return {p: r for p, r in done.items() if r is not None}


@pytest.fixture(scope="module")
def broken(tmp_path_factory, dataset):
    """`minimal` with an unpaced activation offload and no worker stop sentinel."""
    return _run(tmp_path_factory.mktemp("broken"), dataset, "minimal", "broken")


def test_reserved_stays_near_the_allocated_peak(runs, broken):
    assert {"minimal", "low", "balanced"} <= set(runs), sorted(runs)
    for rows, _ in runs.values():
        _bounded(rows)
    must_fail(lambda: _bounded(broken[0]), "an activation offload the host outruns")
    rows = runs["minimal"][0]
    must_fail(
        lambda: _no_creep(_with_creep(rows, 2 * GROWTH)),
        "reserved creeping with no larger batch",
    )


def test_profiles_order_by_device_use(runs):
    device = {p: max(r["vram_device"] for r in rows) for p, (rows, _) in runs.items()}
    _ordered(device)
    swapped = {**device, "minimal": device["balanced"], "balanced": device["minimal"]}
    must_fail(lambda: _ordered(swapped), "profiles out of order")


def test_runs_leave_no_worker_threads(runs, broken):
    for _, left in runs.values():
        _no_threads(left)
    must_fail(lambda: _no_threads(broken[1]), "a worker without its stop sentinel")
