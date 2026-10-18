"""The wheel's kohakumoss/bundle/ (setup.py) carries the presets and user-facing scripts,
never the bench matrix dev package or user configs."""

import importlib.util
import shutil
from pathlib import Path

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
KEPT = ("scripts/train/moss.py", "scripts/sample/sample.py", "configs/moss/_base")
DROPPED = ("scripts/bench/matrix", "configs/user")
OLD_SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "user")


def _setup_module():
    spec = importlib.util.spec_from_file_location("kohakumoss_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_bundle(target: Path) -> None:
    """`target` holds every KEPT path, no DROPPED path and no bytecode."""
    for rel in KEPT:
        assert (target / rel).exists(), f"{rel} missing from the bundle"
    for rel in DROPPED:
        assert not (target / rel).exists(), f"{rel} in the bundle"
    assert not list(target.rglob("*.pyc")), "bytecode in the bundle"


def test_bundle_excludes_the_bench_matrix(tmp_path):
    setup = _setup_module()
    setup.bundle(tmp_path / "bundle")
    check_bundle(tmp_path / "bundle")


def test_negative_the_old_skip_ships_the_bench_matrix(tmp_path):
    setup = _setup_module()
    setup.bundle(tmp_path / "bundle", skip=OLD_SKIP)
    must_fail(
        lambda: check_bundle(tmp_path / "bundle"), "the bench matrix in the bundle"
    )
