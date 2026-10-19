"""Every kogine script with module-level knobs checks them for unknown names (a renamed
knob is an error naming its replacement, never silently ignored): it calls
kohakumoss_common.knobs.check_script, or is one of the scripts with their own validator
built on unknown_problems. check_script itself rejects a name the script does not bind.
"""

import ast
from pathlib import Path

import pytest

from kohakumoss_common.knobs import check_script, declared
from tests._util.negative import must_fail
from tests.scripts.test_entrypoints import imports_argparse

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
OWN_VALIDATOR = {
    "scripts/train/moss.py": "training.knobs.validate via training.entry.run",
    "scripts/sample/sample.py": "inference.knobs.check",
}


def has_knobs(path: Path) -> bool:
    """The file binds an UPPER_CASE name at module level by assignment."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
            isinstance(t, ast.Name) and t.id.isupper() for t in targets
        ):
            return True
    return False


def kogine_scripts(root: Path) -> list[Path]:
    """Scripts under `root`/scripts that document a `kogine run` line, take no argparse
    flags and have module-level knobs."""
    return [
        p
        for p in sorted((root / "scripts").rglob("*.py"))
        if "kogine run scripts/" in p.read_text(encoding="utf-8")
        and not imports_argparse(p)
        and has_knobs(p)
    ]


def unchecked(root: Path) -> list[str]:
    out = []
    for path in kogine_scripts(root):
        rel = path.relative_to(root).as_posix()
        if rel not in OWN_VALIDATOR and "check_script(" not in path.read_text("utf-8"):
            out.append(rel)
    return out


def test_every_kogine_script_checks_its_knobs():
    assert len(kogine_scripts(ROOT)) >= 20, kogine_scripts(ROOT)
    assert unchecked(ROOT) == []


def test_a_script_without_the_check_is_found(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text(
        '"""Demo.\n\n    kogine run scripts/x.py --set FOO=1\n"""\n\nFOO = 0\n\n\n'
        "def main():\n    print(FOO)\n",
        encoding="utf-8",
    )
    must_fail(lambda: _empty(unchecked(tmp_path)), "a script without check_script")


def _empty(items: list) -> None:
    assert items == [], items


def test_check_script_names_the_replacement(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("import os\n\nSAVE_DTYPE = 'bf16'\nOUT = ''\n", encoding="utf-8")
    assert {"os", "SAVE_DTYPE", "OUT"} <= declared(str(script))
    g = {"__file__": str(script), "SAVE_DTYPE": "fp16", "OUT": "x"}
    assert check_script(g) == {"SAVE_DTYPE": "fp16", "OUT": "x"}
    renamed = {"DTYPE": "renamed to SAVE_DTYPE"}
    with pytest.raises(SystemExit, match="unknown knob DTYPE: renamed to SAVE_DTYPE"):
        check_script({**g, "DTYPE": "fp16"}, renamed)
    check_script({**g, "WEIGHTS": None}, accept={"WEIGHTS"})
