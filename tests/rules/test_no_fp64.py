"""No fp64 in the trainer, model, kernels or inference: only reference modules may use it."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
FP64 = re.compile(
    r"float64|\.double\(\)|torch\.double\b|\bfp64\b|\bf64\b|\bF64\b|np\.double\b"
)
REFERENCE = {
    "kohakumoss_kernels/precision.py",
    "kohakumoss_kernels/bench/blockops.py",
    "kohakumoss_kernels/backends/eager/_acc.py",
    "kohakumoss/inference/reference.py",
    "kohakumoss/parallel/selftest.py",
}


def offenders(root: Path) -> dict[str, list[int]]:
    """{relative path: [line numbers]} of fp64 uses outside the reference modules,
    ignoring docstrings and comments."""
    found = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in REFERENCE:
            continue
        lines = []
        in_doc = False
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            quotes = code.count('"""') + code.count("'''")
            if in_doc or code.strip().startswith(('"""', "'''")):
                in_doc = in_doc != (quotes % 2 == 1)
                continue
            if FP64.search(code):
                lines.append(i)
        if lines:
            found[rel] = lines
    return found


def test_no_fp64_outside_references():
    assert offenders(SRC) == {}


def test_check_catches_fp64(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "ok.py").write_text(
        '"""Uses float64 only in prose."""\nx = 1  # float64 note\n'
    )
    (pkg / "bad.py").write_text(
        "import torch\nacc = torch.zeros(4, dtype=torch.float64)\n"
    )
    (pkg / "bad_tl.py").write_text("y = x.to(tl.float64)\n")
    assert offenders(tmp_path) == {"pkg/bad.py": [2], "pkg/bad_tl.py": [1]}
