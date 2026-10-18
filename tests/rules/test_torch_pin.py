"""The torch pin of pyproject.toml is the one every install helper and install doc repeats
(the triton-windows extra pairs with it)."""

import re
from pathlib import Path

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
SPEC_RE = re.compile(r"torch>=[0-9.]+(?:,<[0-9.]+)?")
BOUNDED_RE = re.compile(r"torch>=[0-9.]+,<[0-9.]+")
REPEATS = (
    "scripts/install/install.sh",
    "scripts/install/install.ps1",
    "README.md",
    "docs/guides/install.md",
    "docs/guides/install-windows.md",
)


def pinned(root: Path) -> str:
    """The torch requirement of pyproject.toml's dependencies."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    found = re.search(r'"(torch[<>=!~][^"]*)"', deps)
    assert found is not None, "pyproject.toml pins no torch"
    return found.group(1)


def mismatches(root: Path) -> dict[str, list[str]]:
    """Per repeating file, the torch specs that differ from the pin (a file naming none
    counts as a mismatch)."""
    spec = pinned(root)
    bad = {}
    for rel in REPEATS:
        specs = SPEC_RE.findall((root / rel).read_text(encoding="utf-8"))
        wrong = [s for s in specs if s != spec]
        if wrong or not specs:
            bad[rel] = wrong or ["no torch spec"]
    return bad


def assert_bounded(root: Path) -> None:
    assert BOUNDED_RE.fullmatch(pinned(root)), pinned(root)


def assert_none(found: dict) -> None:
    assert found == {}, found


def test_pin_has_an_upper_bound():
    assert_bounded(ROOT)


def test_every_repeat_equals_the_pin():
    assert_none(mismatches(ROOT))


def _tree(root: Path, pin: str, repeat: str) -> Path:
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'dependencies = [\n    "{pin}",\n    "numpy",\n]\n', encoding="utf-8"
    )
    for rel in REPEATS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'install "{repeat}" torchvision\n', encoding="utf-8")
    return root


def test_check_sees_a_drifted_repeat(tmp_path):
    good = _tree(tmp_path / "good", "torch>=2.13,<2.15", "torch>=2.13,<2.15")
    assert_none(mismatches(good))
    assert_bounded(good)
    drift = _tree(tmp_path / "drift", "torch>=2.13,<2.15", "torch>=2.13,<2.16")
    must_fail(lambda: assert_none(mismatches(drift)), "a helper pinning another torch")
    unbounded = _tree(tmp_path / "unbounded", "torch>=2.13", "torch>=2.13")
    must_fail(lambda: assert_bounded(unbounded), "a pin without an upper bound")
