"""Sample-script knob validation: the known set equals the script, each rule fires on its
broken variant, and the parse helpers read the old flag formats."""

import ast
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kohakumoss.inference.knobs import (
    KNOWN,
    carries_fix,
    check,
    parse_adapter,
    parse_size,
    skip_layers,
    validate,
)
from kohakumoss.model.io.keymap import FP16_FIX_KEY
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sample" / "sample.py"


def script_defaults() -> dict:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        target = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            target, ast.Name
        ):
            if target.id.isupper() and node.value is not None:
                expr = ast.Expression(node.value)
                out[target.id] = eval(compile(expr, str(SCRIPT), "eval"), {})
    return out


def knobs(**overrides) -> dict:
    k = script_defaults()
    k.update(WEIGHTS="weights/root", OUT="out/samples", PROMPT="a cat")
    k.update(overrides)
    return k


def real_problems(k: dict) -> list[str]:
    return [p for p in validate(k) if "does not exist" not in p]


def assert_valid(k: dict) -> None:
    assert real_problems(k) == [], real_problems(k)


def test_known_matches_script():
    assert set(script_defaults()) == KNOWN


def transformer(path: Path, fold: bool) -> str:
    """A single-file transformer whose metadata records an fp16-fix fold or none."""
    meta = {FP16_FIX_KEY: json.dumps({"c": 1 / 16})} if fold else {}
    save_file({"w": torch.zeros(1)}, str(path), meta)
    return str(path)


def test_an_fp16_fix_file_is_known_by_its_metadata(tmp_path):
    fixed = transformer(tmp_path / "fix.safetensors", True)
    assert_valid(knobs(PRECISION="fp16", TRANSFORMER=fixed))
    plain = transformer(tmp_path / "plain.safetensors", False)
    assert any("FP16_FIX" in p for p in real_problems(knobs(PRECISION="fp16")))
    must_fail(
        lambda: assert_valid(knobs(PRECISION="fp16", TRANSFORMER=plain)),
        "a TRANSFORMER file without a recorded fold",
    )
    broken = tmp_path / "broken.safetensors"
    broken.write_text("x")
    assert not carries_fix(str(broken))


def test_valid_knobs_pass():
    assert_valid(knobs())
    assert_valid(
        knobs(PRECISION="fp16", FP16_FIX="online", FP16_FIX_KWARGS={"c": 1 / 16})
    )
    assert_valid(knobs(SIZES=["832x1216", (1024, 1024)], SKIP_LAYERS="22,23"))
    assert_valid(knobs(PROMPT="", PROMPTS=["a", "b"]))
    assert_valid(
        knobs(
            TEXT_ENCODE={"memory": "ring", "free_after_use": True},
            VAE_ENCODE={"memory": "offload", "tile": True},
            MEMORY_PROFILE="minimal",
            BASE_PRECISION="int8",
        )
    )


BROKEN = {
    "unknown knob": dict(STEP=10),
    "PRECISION=": dict(PRECISION="fp8"),
    "VARIANT": dict(VARIANT="turbo"),
    "TEXT_ENCODE": dict(TEXT_ENCODE={"dtype": "int8"}),
    "VAE_ENCODE": dict(VAE_ENCODE={"memory": "gpu"}),
    "FP16_FIX=": dict(FP16_FIX="weights"),
    "CFG_MODE": dict(CFG_MODE="batched"),
    "ADAPTER_MODE": dict(ADAPTER_MODE="add"),
    "MEMORY_PROFILE": dict(MEMORY_PROFILE="fastest"),
    "TUNE": dict(TUNE="always"),
    "WEIGHTS is empty": dict(WEIGHTS=""),
    "OUT is empty": dict(OUT=""),
    "STEPS": dict(STEPS=0),
    "BATCH": dict(BATCH=0),
    "CFG must": dict(CFG=-1.0),
    "needs FP16_FIX_KWARGS": dict(FP16_FIX="online"),
    "PRECISION='fp16' needs": dict(PRECISION="fp16"),
    "multiples of 16": dict(SIZES=[(1000, 1024)]),
    "not 'HxW'": dict(SIZES=["big"]),
    "SKIP_LAYERS": dict(SKIP_LAYERS="all"),
    "no prompts": dict(PROMPT=""),
    "ADAPTERS entry": dict(ADAPTERS=[("a.safetensors", 0.5, 1)]),
}


@pytest.mark.parametrize("needle", BROKEN, ids=list(BROKEN))
def test_each_rule_fires(needle):
    broken = knobs(**BROKEN[needle])
    must_fail(lambda: assert_valid(broken), needle)
    assert any(needle in p for p in validate(broken)), validate(broken)


def test_missing_paths_reported(tmp_path):
    k = knobs(
        WEIGHTS=str(tmp_path / "none"),
        TRANSFORMER=str(tmp_path / "t.safetensors"),
        PROMPTS_FILE=str(tmp_path / "p.txt"),
        ADAPTERS=[str(tmp_path / "a.safetensors") + ":0.8"],
    )
    missing = [p for p in validate(k) if "does not exist" in p]
    assert len(missing) == 4, missing
    for name in ("p.txt", "a.safetensors"):
        (tmp_path / name).write_text("x")
    (tmp_path / "none").mkdir()
    (tmp_path / "t.safetensors").write_text("x")
    assert validate(k) == []


def test_check_raises_once_with_every_problem():
    with pytest.raises(SystemExit) as err:
        check(knobs(STEPS=0, BATCH=0, STEPZ=1))
    msg = str(err.value)
    for needle in ("STEPS", "BATCH", "did you mean STEPS"):
        assert needle in msg, needle


def test_parse_size():
    assert parse_size("832x1216") == (832, 1216)
    assert parse_size("1024X768") == (1024, 768)
    assert parse_size((512, 640)) == (512, 640)
    assert parse_size([512, 640]) == (512, 640)
    with pytest.raises(ValueError):
        parse_size("1024")


def test_parse_adapter():
    assert parse_adapter("lora.safetensors") == ("lora.safetensors", 1.0)
    assert parse_adapter("lora.safetensors:0.8") == ("lora.safetensors", 0.8)
    assert parse_adapter("lora.safetensors:-1") == ("lora.safetensors", -1.0)
    assert parse_adapter("C:\\a.safetensors") == ("C:\\a.safetensors", 1.0)
    assert parse_adapter("C:\\a.safetensors:0.5") == ("C:\\a.safetensors", 0.5)
    assert parse_adapter(("x.safetensors", 0.3)) == ("x.safetensors", 0.3)


def test_skip_layers():
    assert skip_layers("default") is None
    assert skip_layers("none") == ()
    assert skip_layers("22,23") == (22, 23)
    assert skip_layers("22") == (22,)
    assert skip_layers((22,)) == (22,)
    assert skip_layers([]) == ()
