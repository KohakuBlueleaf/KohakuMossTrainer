"""Every script imports from the installed packages alone: no repo root, no script dir.
Modules of a dev package under scripts/ (a directory with `__init__.py`, e.g. the bench
matrix's `kohakumoss_bench`) are package modules, not scripts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    p
    for p in (ROOT / "scripts").rglob("*.py")
    if not p.name.startswith("_") and not (p.parent / "__init__.py").exists()
)
CHECK = """
import json, runpy, sys, traceback
bad = {}
for path in sys.argv[1:]:
    try:
        runpy.run_path(path, run_name="__import_check__")
    except BaseException:
        bad[path] = traceback.format_exc(limit=3)[-600:]
print(json.dumps(bad))
"""


def import_failures(paths: list[Path], cwd: Path) -> dict[str, str]:
    """Scripts that fail to import in a fresh interpreter whose cwd is `cwd`."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [sys.executable, "-c", CHECK, *map(str, paths)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_scripts_import_without_path_setup(tmp_path):
    assert len(SCRIPTS) >= 20, SCRIPTS
    assert import_failures(SCRIPTS, tmp_path) == {}


def test_check_catches_a_sibling_import(tmp_path):
    (tmp_path / "_helper.py").write_text("X = 1\n")
    broken = tmp_path / "uses_sibling.py"
    broken.write_text("from _helper import X\n")
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    assert list(import_failures([broken], elsewhere)) == [str(broken)]
