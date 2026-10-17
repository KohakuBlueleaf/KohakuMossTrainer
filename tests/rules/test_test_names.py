"""Test modules outside packages have unique basenames (pytest's prepend import mode)."""

from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]


def clashes(tests_root: Path) -> dict[str, list[str]]:
    """{module basename: [paths]} for test modules imported by basename (no __init__.py
    in their directory) that collide with another test module's basename."""
    by_name = defaultdict(list)
    for path in sorted(tests_root.rglob("*.py")):
        if path.name.startswith(("test_", "_")) or path.name == "conftest.py":
            packaged = (path.parent / "__init__.py").exists()
            key = path.stem if not packaged else None
            if key is not None and path.name != "__init__.py":
                by_name[key].append(path.relative_to(tests_root).as_posix())
    return {k: v for k, v in by_name.items() if len(v) > 1 and k != "conftest"}


def test_no_basename_clashes():
    assert clashes(ROOT / "tests") == {}


def test_check_catches_a_clash(tmp_path):
    for area in ("a", "b"):
        (tmp_path / area).mkdir()
        (tmp_path / area / "test_same.py").write_text("")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "__init__.py").write_text("")
    (tmp_path / "c" / "test_same.py").write_text("")
    assert clashes(tmp_path) == {"test_same": ["a/test_same.py", "b/test_same.py"]}
