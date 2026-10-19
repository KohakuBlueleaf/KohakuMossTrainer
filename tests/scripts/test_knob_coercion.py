"""check_script's owner knobs: a string value of a knob a script takes from its owner
scripts is coerced against that owner's default or annotation, as kogine coerces a
`--set` value of a bound knob; a string knob no owner types is a knob error. End to end,
`kogine run scripts/data/cache_text.py --set TARGET_AREA=262144` reaches the script as
the int, through kogine's own CLI parse. An owner's schema is read from its source and
equals kogine's import of it for every knob the source types."""

import ast
import operator
import sys
from pathlib import Path

import pytest
from kohakuengine import cli as kogine_cli
from kohakuengine import introspect_schema

from kohakumoss_common import knobs, script_schema
from kohakumoss_common.knobs import check_script, declared
from kohakumoss_common.script_schema import static_schema
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "scripts" / "train" / "moss.py"
OWNERS = ("../train/moss.py", ".")
OWNER_FILES = [TRAIN, *sorted((ROOT / "scripts" / "data").glob("*.py"))]
IMPORTED = {"moss.py": {"SAMPLE_PROMPTS"}}
GIVEN = {
    "CACHE_DIR": "c",
    "TARGET_AREA": "262144",
    "ASPECT_RANGE": "(0.5, 2)",
    "METRICS_EVERY": "none",
    "UPSCALE": "to_area",
    "ENCODE_BATCH": "64",
    "CAPTION": {"template": "x"},
}
COERCED = {
    "CACHE_DIR": "c",
    "TARGET_AREA": 262144,
    "ASPECT_RANGE": (0.5, 2.0),
    "METRICS_EVERY": None,
    "UPSCALE": "to_area",
    "ENCODE_BATCH": 64,
    "CAPTION": {"template": "x"},
}


def _pass_through(k, foreign, binds):
    return {}, []


def owner_tree(tmp_path: Path) -> Path:
    """A train script owning typed knobs, a sibling owning one more, and a data script
    binding only CACHE_DIR; returns the data script."""
    (tmp_path / "train").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "train" / "moss.py").write_text(
        "TARGET_AREA = 1024 * 1024\nASPECT_RANGE = (0.25, 4.0)\n"
        "METRICS_EVERY: int | None = None\nUPSCALE = 'never'\n"
        "CAPTION = {'template': 'tags'}\n",
        encoding="utf-8",
    )
    (tmp_path / "data" / "other.py").write_text("ENCODE_BATCH = 512\n", "utf-8")
    script = tmp_path / "data" / "cache.py"
    script.write_text("CACHE_DIR = ''\n", encoding="utf-8")
    return script


def check_coerced(script: Path) -> None:
    g = {"__file__": str(script), **GIVEN}
    k = check_script(g, owners=OWNERS)
    assert k == COERCED, k
    assert {n: g[n] for n in COERCED} == COERCED, g
    assert type(g["TARGET_AREA"]) is int and type(g["ENCODE_BATCH"]) is int, g
    assert all(type(v) is float for v in g["ASPECT_RANGE"]), g


def test_owner_strings_are_coerced(tmp_path, monkeypatch):
    script = owner_tree(tmp_path)
    check_coerced(script)
    monkeypatch.setattr(knobs, "owned_values", _pass_through)
    must_fail(lambda: check_coerced(script), "check_script without the coercion")


def test_failed_coercion_keeps_the_string(tmp_path):
    script = owner_tree(tmp_path)
    g = {"__file__": str(script), "TARGET_AREA": "abc"}
    with pytest.warns(UserWarning, match="TARGET_AREA"):
        k = check_script(g, owners=OWNERS)
    assert k["TARGET_AREA"] == g["TARGET_AREA"] == "abc"


def check_untyped_refused(script: Path) -> None:
    g = {"__file__": str(script), "WEIGHTS": "w"}
    with pytest.raises(SystemExit, match="WEIGHTS='w': no owner script binds"):
        check_script(g, accept={"WEIGHTS"}, owners=OWNERS)


def test_untyped_string_knob_is_an_error(tmp_path, monkeypatch):
    script = owner_tree(tmp_path)
    check_untyped_refused(script)
    g = {"__file__": str(script), "WEIGHTS": 3}
    assert check_script(g, accept={"WEIGHTS"}, owners=OWNERS)["WEIGHTS"] == 3
    with pytest.raises(SystemExit, match="unknown knob TARGET_AREAS"):
        check_script({**g, "TARGET_AREAS": 1}, accept={"WEIGHTS"}, owners=OWNERS)
    monkeypatch.setattr(knobs, "owned_values", _pass_through)
    must_fail(lambda: check_untyped_refused(script), "untyped strings let through")


def check_owners_typed(files: list[Path]) -> None:
    untyped = {}
    for path in files:
        names = {n for n in declared(str(path)) if n.isupper()}
        missing = names - introspect_schema(path).declared
        if missing:
            untyped[path.name] = sorted(missing)
    assert untyped == {}, untyped


def test_every_owner_knob_is_typed(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    check_owners_typed(OWNER_FILES)
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n\nJOIN = os.path.join\n", encoding="utf-8")
    must_fail(lambda: check_owners_typed([bad]), "an owner knob bound to an import")


def _shown(value) -> tuple:
    return type(value), repr(value)


def static_gaps(path: Path) -> tuple[list[str], set[str]]:
    """(knobs whose source-read default or annotation differs from kogine's
    introspect_schema, knobs the source leaves to the import) of the owner `path`."""
    found = static_schema(str(path))
    assert found is not None, f"{path.name}: not plain declarations"
    schema, opaque = found
    full = introspect_schema(path)
    upper = {n for n in full.declared | schema.declared if n.isupper()}
    gaps = []
    for name in sorted(upper - opaque):
        for part in ("defaults", "annotations"):
            ours, theirs = getattr(schema, part), getattr(full, part)
            if (name in ours, _shown(ours.get(name))) != (
                name in theirs,
                _shown(theirs.get(name)),
            ):
                gaps.append(f"{path.name}:{name} {part}")
    return gaps, {n for n in opaque if n.isupper()}


def check_static_agrees(files: list[Path], imported: dict[str, set[str]]) -> None:
    for path in files:
        gaps, opaque = static_gaps(path)
        assert gaps == [], gaps
        assert opaque == imported.get(path.name, set()), (path.name, opaque)


def test_owner_source_schema_is_kogines(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    check_static_agrees(OWNER_FILES, IMPORTED)
    wrong = {**script_schema.BINARY, ast.Mult: operator.add}
    monkeypatch.setattr(script_schema, "BINARY", wrong)
    must_fail(lambda: check_static_agrees([TRAIN], IMPORTED), "1024 * 1024 misread")


def test_source_schema_shapes(tmp_path):
    script = tmp_path / "s.py"
    script.write_text(
        '"""Doc."""\n\nimport os\nfrom typing import Annotated\n\nA = 2 ** 10\n'
        "B: tuple[int, int] | None = (A, -A)\nC = os.sep\nD = {'k': [A // 3, 1 / 4]}\n"
        "E: Annotated[int, 0] = 1\nC = 'x'\nF = D['k'][0]\nG = 1\n"
        "from os import sep as G\nH: 'int | None' = 2\n\n\ndef main():\n"
        "    return A\n\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    schema, opaque = static_schema(str(script))
    expected = {"A": 1024, "B": (1024, -1024), "C": "x", "D": {"k": [341, 0.25]}}
    assert schema.defaults == {**expected, "E": 1, "F": 341, "H": 2}, schema.defaults
    hints = {"B": tuple[int, int] | None, "H": int | None}
    assert schema.annotations == hints, schema.annotations
    assert opaque == {"os", "Annotated", "main", "E", "G"}, opaque
    for body in (
        "X = 1\nif X:\n    X = 2\n",
        "X = []\nX.append(1)\n",
        "X = 1\nX += 1\n",
    ):
        script.write_text(body, encoding="utf-8")
        assert static_schema(str(script)) is None, body


def _no_import(path):
    raise AssertionError(f"imported {path}")


def check_parsed_only(monkeypatch) -> None:
    script = ROOT / "scripts" / "data" / "cache_text.py"
    g = {"__file__": str(script), "TARGET_AREA": "262144", "SAMPLE_CFG": "4"}
    with monkeypatch.context() as m:
        m.setattr(knobs, "introspect_schema", _no_import)
        k = check_script(g, owners=OWNERS)
    assert k["TARGET_AREA"] == 262144 and k["SAMPLE_CFG"] == 4.0, k
    assert type(k["SAMPLE_CFG"]) is float, k


def test_owner_knobs_are_read_without_an_import(monkeypatch):
    check_parsed_only(monkeypatch)
    script = ROOT / "scripts" / "data" / "cache_text.py"
    g = {"__file__": str(script), "SAMPLE_PROMPTS": "[a, b]"}
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert check_script(g, owners=OWNERS)["SAMPLE_PROMPTS"] == ["a", "b"]
    monkeypatch.setattr(knobs, "static_schema", lambda path: None)
    must_fail(lambda: check_parsed_only(monkeypatch), "every owner imported")


class Stop(Exception):
    """Raised after the script's knob check, before it builds anything."""


def cli_knobs(monkeypatch, *sets: str) -> dict:
    """The globals of scripts/data/cache_text.py after its check_script, run through
    `kogine run ... --set`; the script stops there."""
    seen = {}
    real = knobs.check_script

    def recording(g, *args, **kwargs):
        real(g, *args, **kwargs)
        seen.update(g)
        raise Stop

    script = str(ROOT / "scripts" / "data" / "cache_text.py")
    args = [arg for s in sets for arg in ("--set", s)]
    with monkeypatch.context() as m:
        m.setattr(knobs, "check_script", recording)
        m.setattr(sys, "path", list(sys.path))
        with pytest.raises(Stop):
            kogine_cli.main(["run", script, *args])
    return seen


def check_cli_int(monkeypatch) -> None:
    seen = cli_knobs(monkeypatch, "TARGET_AREA=262144", "CACHE_DTYPE=fp16")
    assert type(seen["TARGET_AREA"]) is int and seen["TARGET_AREA"] == 262144, seen
    assert seen["CACHE_DTYPE"] == "fp16", seen


@pytest.mark.budget(60)
def test_kogine_set_reaches_cache_text_typed(monkeypatch):
    check_cli_int(monkeypatch)
    monkeypatch.setattr(knobs, "owned_values", _pass_through)
    must_fail(lambda: check_cli_int(monkeypatch), "the CLI run without the coercion")
