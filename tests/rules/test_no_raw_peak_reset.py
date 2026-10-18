"""No raw peak-counter reset or read outside the process-wide peak accumulator
(`kohakumoss_kernels/hardware/memory.py`): a raw `reset_peak_memory_stats` cuts every
enclosing measurement, and a raw `max_memory_*` read is cut by everyone else's windows.
Sources, scripts and tests are checked; defining a fake allocator's methods is fine."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
TREES = ("src", "scripts", "tests")
RAW = re.compile(
    r"\.(reset_peak_memory_stats|max_memory_allocated|max_memory_reserved)\("
)
OWNER = "src/kohakumoss_kernels/hardware/memory.py"
ALLOWED = {
    OWNER,
    "tests/training/memory/test_vram_accounting.py",
    "tests/rules/test_no_raw_peak_reset.py",
}


def offenders(root: Path, trees=TREES) -> dict[str, list[int]]:
    """{relative path: [line numbers]} of raw peak-counter calls outside ALLOWED,
    comments ignored."""
    found = {}
    for tree in trees:
        for path in sorted((root / tree).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel in ALLOWED:
                continue
            lines = [
                i
                for i, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1
                )
                if RAW.search(line.split("#", 1)[0])
            ]
            if lines:
                found[rel] = lines
    return found


def test_no_raw_peak_counter_use():
    assert offenders(ROOT) == {}


def test_check_catches_a_raw_reset(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "ok.py").write_text(
        "class Fake:\n    def reset_peak_memory_stats(self, d):\n        pass\n"
        "x = 1  # torch.cuda.reset_peak_memory_stats() in a comment\n"
    )
    (src / "bad.py").write_text(
        "import torch\ntorch.cuda.reset_peak_memory_stats(0)\n"
        "peak = torch.cuda.max_memory_allocated(0)\n"
    )
    assert offenders(tmp_path, ("src",)) == {"src/bad.py": [2, 3]}
