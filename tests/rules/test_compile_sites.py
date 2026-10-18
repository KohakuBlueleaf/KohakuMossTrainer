"""Every torch.compile site in src/ and scripts/ runs the process's compile settings
first: its enclosing function calls `compile_setup.apply()` before the compile call (a
module's `.compile(...)` counts as a site). The check finds a constructed site without
it."""

import ast
from pathlib import Path

import pytest

from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
TREES = (ROOT / "src", ROOT / "scripts")
NOT_TORCH = {"re", "regex", "tilelang"}
COMPILE_KWARGS = {"dynamic", "mode", "fullgraph", "backend", "options"}


def is_compile(node: ast.Call) -> bool:
    """`torch.compile(...)`, or `<module>.compile(...)` with a torch.compile keyword."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "compile":
        return False
    if isinstance(func.value, ast.Name) and func.value.id == "torch":
        return True
    owner = func.value.id if isinstance(func.value, ast.Name) else ""
    keywords = {k.arg for k in node.keywords}
    return owner not in NOT_TORCH and bool(keywords & COMPILE_KWARGS)


def is_apply(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "apply"
        and isinstance(func.value, ast.Name)
        and func.value.id == "compile_setup"
    )


def unguarded(root: Path) -> list[str]:
    """`path:line` of every compile site whose enclosing function makes no
    compile_setup.apply() call before it."""
    out = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "compile_setup.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            applied = [n.lineno for n in calls if is_apply(n)]
            for call in calls:
                if is_compile(call) and not any(a <= call.lineno for a in applied):
                    out.append(f"{path.relative_to(root).as_posix()}:{call.lineno}")
        top = [n for n in tree.body if isinstance(n, ast.Expr | ast.Assign)]
        for node in top:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and is_compile(call):
                    out.append(f"{path.relative_to(root).as_posix()}:{call.lineno}")
    return sorted(set(out))


def _none(items: list[str]) -> None:
    assert items == [], items


def test_every_compile_site_applies_the_settings_first():
    _none([site for tree in TREES for site in unguarded(tree)])


def test_a_site_without_the_settings_is_found(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "good.py").write_text(
        "import torch\nfrom k import compile_setup\n\n\ndef f(fn):\n"
        "    compile_setup.apply()\n    return torch.compile(fn)\n",
        encoding="utf-8",
    )
    (pkg / "bad.py").write_text(
        "import re\n\n\ndef g(model):\n    re.compile('x')\n"
        "    model.compile(dynamic=True)\n",
        encoding="utf-8",
    )
    assert unguarded(tmp_path) == ["pkg/bad.py:6"]
    must_fail(lambda: _none(unguarded(tmp_path)), "a compile site without the setup")
