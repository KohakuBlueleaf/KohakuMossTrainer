"""compare_grid.py declares every knob it reads: the MEMORY_PROFILE='manual' placement
knobs, as sample.py does, so check_script accepts a manual-placement grid config."""

from pathlib import Path

import pytest

from kohakumoss.memory.knob_map import PLACEMENT_KNOBS
from kohakumoss_common.knobs import declared
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SAMPLE = Path(__file__).resolve().parents[2] / "scripts" / "sample"


def _declares_placement(path: Path) -> None:
    missing = set(PLACEMENT_KNOBS) - declared(str(path))
    assert not missing, (path.name, missing)


def test_grid_declares_the_manual_placement_knobs(tmp_path):
    for name in ("compare_grid.py", "sample.py"):
        _declares_placement(SAMPLE / name)
    text = (SAMPLE / "compare_grid.py").read_text(encoding="utf-8")
    kept = [line for line in text.splitlines() if not line.startswith("STREAM")]
    stripped = tmp_path / "compare_grid.py"
    stripped.write_text("\n".join(kept) + "\n", encoding="utf-8")
    must_fail(lambda: _declares_placement(stripped), "the STREAM knobs undeclared")
