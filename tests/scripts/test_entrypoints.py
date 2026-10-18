"""Every `kogine run` script resolves to an entrypoint kogine can call with no arguments;
user-facing scripts take knobs, not argparse flags; documented usage names real knobs.
"""

import ast
import importlib.util
import inspect
import re
from pathlib import Path

import pytest
from kohakuengine import introspect_schema
from kohakuengine.engine.entrypoint import find_entrypoint

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted(
    p
    for p in (ROOT / "scripts").rglob("*.py")
    if "kogine run scripts/" in p.read_text(encoding="utf-8")
)
ARGPARSE_DIRS = ("scripts/bench/", "scripts/colab/")
RUN_LINE = re.compile(r"kogine run (scripts/\S+\.py)(.*)")
SET_KEY = re.compile(r"--set\s+([A-Za-z_]\w*)=")
CONFIG_PATH = re.compile(r"--config\s+(\S+)")


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


def imports_argparse(path: Path) -> bool:
    """True when the file imports argparse anywhere (any scope)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "argparse" for a in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "argparse":
                return True
    return False


def user_scripts(root: Path) -> list[Path]:
    """Scripts under `root`/scripts that must take knobs instead of flags."""
    out = []
    for path in sorted((root / "scripts").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(ARGPARSE_DIRS):
            continue
        out.append(path)
    return out


def run_lines(path: Path) -> list[tuple[str, str]]:
    """(script, rest of line) of every `kogine run` line in the module docstring, with
    backslash continuations joined."""
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    doc = re.sub(r"\\\s*\n\s*", " ", doc)
    return [m.groups() for m in RUN_LINE.finditer(doc)]


def undeclared_keys(path: Path, root: Path) -> list[str]:
    """`--set KEY=` keys of `path`'s usage lines the named script does not declare."""
    out = []
    for script, rest in run_lines(path):
        declared = introspect_schema(root / script).declared
        out += [f"{script}: {k}" for k in SET_KEY.findall(rest) if k not in declared]
    return out


def missing_configs(path: Path, root: Path) -> list[str]:
    """`--config <path>` entries of `path`'s usage lines that do not exist (placeholders
    in <> skipped)."""
    out = []
    for _, rest in run_lines(path):
        for cfg in CONFIG_PATH.findall(rest):
            if "<" not in cfg and not (root / cfg).is_file():
                out.append(cfg)
    return out


def assert_no_argparse(paths: list[Path]) -> None:
    bad = [p.name for p in paths if imports_argparse(p)]
    assert bad == [], f"argparse in user scripts: {bad}"


def test_scripts_found():
    assert len(SCRIPTS) >= 12, SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_entrypoint_takes_no_arguments(path):
    fn = find_entrypoint(load(path), path)
    assert required_params(fn) == [], f"{path.name}: kogine calls {fn.__name__}()"


def test_user_scripts_take_knobs():
    scripts = user_scripts(ROOT)
    assert len(scripts) >= 15, scripts
    assert_no_argparse(scripts)


def test_argparse_check_catches_a_flag_script(tmp_path):
    (tmp_path / "scripts" / "probe").mkdir(parents=True)
    bad = tmp_path / "scripts" / "probe" / "flags.py"
    bad.write_text("def main():\n    from argparse import ArgumentParser\n")
    must_fail(lambda: assert_no_argparse(user_scripts(tmp_path)), "argparse script")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_documented_usage_is_real(path):
    assert run_lines(path), f"{path.name}: no kogine run line"
    assert undeclared_keys(path, ROOT) == []
    assert missing_configs(path, ROOT) == []


def test_usage_check_catches_unknown_key_and_config(tmp_path):
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "x.py"
    script.write_text(
        '"""Demo.\n\n    kogine run scripts/x.py --set FOO=1 \\\n'
        '        --set NOPE=2 --config configs/none.py\n"""\n\nFOO = 0\n'
    )
    assert undeclared_keys(script, tmp_path) == ["scripts/x.py: NOPE"]
    must_fail(lambda: _assert_empty(undeclared_keys(script, tmp_path)), "unknown key")
    must_fail(lambda: _assert_empty(missing_configs(script, tmp_path)), "no config")


def _assert_empty(items: list) -> None:
    assert items == [], items
