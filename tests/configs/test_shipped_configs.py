"""Every config under configs/ names the script it drives and validates (paths aside)
through that script's knob checks: script defaults, its base config, then the file."""

import sys
from pathlib import Path

import pytest
import torch

from kohakumoss.cli.config.validate import (
    blocking,
    config_problems,
    shipped_configs,
    target_script,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = shipped_configs(ROOT)


@pytest.fixture
def linux_cpu(monkeypatch):
    """Device- and platform-independent validation (multi-GPU layers are Linux-only)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")


def assert_valid(path: Path) -> None:
    bad = blocking(config_problems(path, ROOT))
    assert bad == [], bad


def test_configs_found():
    groups = {p.parent.relative_to(ROOT / "configs").as_posix() for p in CONFIGS}
    for group in ("moss/algo", "moss/vram", "moss/precision", "moss/parallel"):
        assert group in groups, groups
    for group in ("moss/data", "moss/recipes", "sample/precision"):
        assert group in groups, groups


@pytest.mark.parametrize(
    "path", CONFIGS, ids=lambda p: p.relative_to(ROOT / "configs").as_posix()
)
def test_config_validates(path, linux_cpu):
    assert (ROOT / target_script(path)).is_file()
    assert_valid(path)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_check_catches_broken_configs(tmp_path, linux_cpu):
    base = ROOT / "configs" / "moss" / "_base" / "base_lora.py"
    head = '"""Drives scripts/train/moss.py."""\nfrom kohakuengine import use_config\n'
    head += f'use_config("{base.as_posix()}")\n'
    good = _write(tmp_path, "good.py", head + "DIM = 8\n")
    assert_valid(good)
    for name, body in (
        ("typo.py", "DIMM = 8\n"),
        ("choice.py", 'MEMORY_PROFILE = "fastest"\n'),
        ("conflict.py", 'PARALLEL = "fsdp"\nOPTIMIZER = "muon"\nDEVICES = (0, 1)\n'),
        ("data.py", 'RESOLUTION_MODE = "stretch"\n'),
    ):
        path = _write(tmp_path, name, head + body)
        must_fail(lambda p=path: assert_valid(p), name)
    sample = _write(
        tmp_path,
        "sample.py",
        '"""Drives scripts/sample/sample.py."""\nDTYPE = "fp16"\n',
    )
    must_fail(lambda: assert_valid(sample), "fp16 sampling without a fix")


def test_config_must_name_its_script(tmp_path):
    path = _write(tmp_path, "anon.py", '"""No script named."""\nDIM = 8\n')
    with pytest.raises(ValueError):
        target_script(path)
