"""`kohakumoss-config new --from <base>` and `wizard`: the written file loads through
kogine's loader to the composed knobs, extends its base by use_config, and lists every
knob of scripts/train/moss.py (active or commented)."""

import re
import sys
from pathlib import Path

import pytest
import torch
from kohakuengine import load_config_file

from kohakumoss.cli.config import full, validate
from kohakumoss.cli.config import main as cli
from kohakumoss.cli.config.validate import TRAIN, compose, config_knobs, script_defaults
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
RECIPES = cli.recipes(ROOT)
KNOB_LINE = re.compile(r"^(?:# )?([A-Z][A-Z0-9_]*) = ", re.M)


@pytest.fixture(autouse=True)
def linux_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")


def assert_every_knob(text: str) -> None:
    listed = set(KNOB_LINE.findall(text))
    missing = set(script_defaults(ROOT, TRAIN)) - listed
    assert not missing, sorted(missing)


@pytest.mark.parametrize("recipe", RECIPES)
def test_from_recipe(tmp_path, recipe):
    out = tmp_path / "my_run.py"
    cli.main(["new", "--from", recipe, str(out)])
    text = out.read_text(encoding="utf-8")
    text.encode("ascii")
    assert_every_knob(text)
    base = cli.base_path(ROOT, recipe)
    assert f'use_config("{base.as_posix()}")' in text
    assert load_config_file(out).globals_dict.keys() >= {"WEIGHTS", "DATASETS"}
    assert config_knobs(out, ROOT)[1] == compose(ROOT, TRAIN, [base], {})


@pytest.mark.parametrize("name", ["base_full", "recipes_dora"])
def test_base_names(name):
    assert cli.base_path(ROOT, name).stem == name


def test_listing_check_fails_on_a_dropped_knob(tmp_path, monkeypatch):
    out = tmp_path / "a.py"
    entries = full.knob_entries(ROOT / TRAIN)
    monkeypatch.setattr(
        full, "knob_entries", lambda path: [e for e in entries if e[1] != "LR"]
    )
    cli.main(["new", "--from", "lora_24g_bf16", str(out)])
    must_fail(lambda: assert_every_knob(out.read_text(encoding="utf-8")), "no LR")


def test_knob_comments():
    entries = {e[1]: e[2] for e in full.knob_entries(ROOT / TRAIN) if e[0] == "knob"}
    assert entries["VARIANT"].startswith('"base"'), entries["VARIANT"]
    assert "HF dir" in entries["WEIGHTS"]
    sections = [e for e in full.knob_entries(ROOT / TRAIN) if e[0] == "section"]
    assert any("MODEL" in s[1] for s in sections)


def test_from_refuses_unknown_and_foreign_bases(tmp_path):
    with pytest.raises(SystemExit, match="no base config"):
        cli.main(["new", "--from", "nope", str(tmp_path / "x.py")])
    sample = ROOT / "configs" / "sample" / "_base" / "base_sample.py"
    with pytest.raises(SystemExit, match="does not drive"):
        cli.main(["new", "--from", str(sample), str(tmp_path / "x.py")])


def test_bundle_root(tmp_path, monkeypatch):
    """A wheel install (no checkout above the package) resolves kohakumoss/bundle/."""
    package = tmp_path / "site" / "kohakumoss"
    for sub in ("configs", "scripts"):
        (package / "bundle" / sub).mkdir(parents=True)
    monkeypatch.setattr(validate.kohakumoss, "__file__", str(package / "__init__.py"))
    root = validate.repo_root()
    assert root == package / "bundle" and not validate.is_checkout(root)
    assert validate.script_ref(root, TRAIN) == (root / TRAIN).as_posix()
    assert validate.is_checkout(ROOT) and validate.script_ref(ROOT, TRAIN) == TRAIN
    (package / "bundle" / "scripts").rmdir()
    with pytest.raises(SystemExit, match="not found"):
        validate.repo_root()


def test_wizard(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "d").mkdir()
    out = tmp_path / "wiz.py"
    answers = iter(
        ["12", "fp16", "lokr", "kohya", str(tmp_path / "d"), str(tmp_path / "w")]
        + ["300", "1", "wiz", str(out)]
    )
    cli.wizard(ROOT, ask=lambda prompt: next(answers))
    text = out.read_text(encoding="utf-8")
    assert_every_knob(text)
    _, k = config_knobs(out, ROOT)
    assert (k["ALGO"], k["PRECISION"], k["STEPS"]) == ("lokr", "fp16", 300)
    assert k["MEMORY_PROFILE"] == "ckpt-stream4-actoff-optstream-gradoff"
    assert k["RUN_NAME"] == "wiz"
    assert (tmp_path / "wiz_data.py").is_file()


def test_wizard_reasks_bad_choices(tmp_path):
    out = tmp_path / "w2.py"
    answers = iter(
        ["24", "fp32", "bf16", "lorax", "", "kohya", "d", "w", "", "", "w2", str(out)]
    )
    cli.wizard(ROOT, ask=lambda prompt: next(answers))
    _, k = config_knobs(out, ROOT)
    assert k["ALGO"] == "lora" and k["PRECISION"] == "bf16"
