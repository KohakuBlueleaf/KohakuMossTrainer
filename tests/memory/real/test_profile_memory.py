"""Real-weights memory profiles: over a short LoRA run on a slice of the
KOHAKUMOSS_DATASET dataset (bucketed sizes, so the token count varies), the allocator's
reserved memory stays within a bound of the allocated peak (no growth from step to
step); the profiles order by this process's device use (what nvidia-smi shows):
minimal < low < balanced < speed, for every profile that fits the card; and a finished
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
GROWTH = 0.25  # GiB reserved may grow from step 2 to the last step


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


def _bounded(rows: list[dict]) -> None:
    peak = max(r["vram_peak_alloc"] for r in rows)
    held = max(r["vram_peak_reserved"] for r in rows)
    assert held <= SLACK * peak + PAGES, (held, peak)
    grown = rows[-1]["vram_peak_reserved"] - rows[1]["vram_peak_reserved"]
    assert grown <= GROWTH, (grown, [r["vram_peak_reserved"] for r in rows])


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


def test_profiles_order_by_device_use(runs):
    device = {p: max(r["vram_device"] for r in rows) for p, (rows, _) in runs.items()}
    _ordered(device)
    swapped = {**device, "minimal": device["balanced"], "balanced": device["minimal"]}
    must_fail(lambda: _ordered(swapped), "profiles out of order")


def test_runs_leave_no_worker_threads(runs, broken):
    for _, left in runs.values():
        _no_threads(left)
    must_fail(lambda: _no_threads(broken[1]), "a worker without its stop sentinel")
