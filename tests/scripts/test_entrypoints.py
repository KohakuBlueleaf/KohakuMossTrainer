"""Every `kogine run` script resolves to an entrypoint kogine can call with no arguments."""

import importlib.util
import inspect
from pathlib import Path

import pytest
from kohakuengine.engine.entrypoint import find_entrypoint

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    p
    for p in (ROOT / "scripts").rglob("*.py")
    if "kogine run scripts/" in p.read_text(encoding="utf-8")
)


def required_params(fn) -> list[str]:
    """Parameters of `fn` without a default (kogine passes none for a bare config)."""
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_kogine_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scripts_found():
    assert len(SCRIPTS) >= 5, SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_entrypoint_takes_no_arguments(path):
    fn = find_entrypoint(load(path), path)
    assert required_params(fn) == [], f"{path.name}: kogine calls {fn.__name__}()"
