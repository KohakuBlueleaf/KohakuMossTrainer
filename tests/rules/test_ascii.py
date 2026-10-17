"""Python source in every tracked area is ASCII-only (Windows consoles with a non-UTF-8 code page)."""

from pathlib import Path

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
AREAS = ("src", "scripts", "tests", "configs")


def non_ascii(text: str) -> list[int]:
    """1-based numbers of the lines holding a non-ASCII character."""
    return [i + 1 for i, line in enumerate(text.splitlines()) if not line.isascii()]


def offenders(root: Path) -> dict[str, list[int]]:
    bad = {}
    for area in AREAS:
        for path in sorted((root / area).rglob("*.py")):
            lines = non_ascii(path.read_text(encoding="utf-8"))
            if lines:
                bad[path.relative_to(root).as_posix()] = lines
    return bad


def test_python_source_is_ascii():
    assert offenders(ROOT) == {}


def test_check_sees_a_non_ascii_comment(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("x = 1\n")
    (tmp_path / "src" / "bad.py").write_text("x = 1\n# delta " + chr(0x394) + "W\n")
    must_fail(lambda: _assert_empty(offenders(tmp_path)), "a non-ASCII comment")
    assert offenders(tmp_path) == {"src/bad.py": [2]}


def _assert_empty(found: dict) -> None:
    assert found == {}, found
