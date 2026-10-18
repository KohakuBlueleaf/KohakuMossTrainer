"""A test module that imports the bench dev package (`kohakumoss_bench`, not in the wheel)
skips through `pytest.importorskip` before its first import of it, so the suite collects
where the dev package is not installed."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
DEV = "kohakumoss_bench"


def _imports_dev(node: ast.stmt) -> bool:
    match node:
        case ast.Import(names=names):
            return any(a.name.split(".")[0] == DEV for a in names)
        case ast.ImportFrom(module=module):
            return (module or "").split(".")[0] == DEV
    return False


def _skips(node: ast.stmt) -> bool:
    match node:
        case ast.Expr(value=ast.Call(func=ast.Attribute(attr="importorskip"), args=a)):
            return bool(a) and getattr(a[0], "value", None) == DEV
    return False


def unguarded(root: Path) -> list[str]:
    """Test modules whose first module-level import of the dev package comes before an
    `importorskip` of it."""
    found = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        guarded = False
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if _skips(node):
                guarded = True
            elif _imports_dev(node):
                if not guarded:
                    found.append(path.relative_to(root).as_posix())
                break
    return found


def test_dev_package_imports_are_guarded():
    assert unguarded(ROOT) == []


def test_check_catches_an_unguarded_import(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text(
        f'import pytest\npytest.importorskip("{DEV}")\nfrom {DEV}.report import x\n'
    )
    (tests / "test_bad.py").write_text(f"import pytest\nfrom {DEV}.report import x\n")
    assert unguarded(tmp_path) == ["tests/test_bad.py"]
