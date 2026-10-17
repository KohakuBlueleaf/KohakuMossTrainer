"""Entry features end to end on the tiny model: Lightning driver, sample callback,
validation, every logger, doctor report. Runs on Windows and Linux, CPU or GPU."""

import importlib.util
import json
import sys

import pytest

from kohakumoss.cli import doctor
from kohakumoss.training.entry import run
from tests.training.test_entry_smoke import smoke_knobs

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def test_lightning_driver_entry(tmp_path):
    state = run(smoke_knobs(tmp_path, DRIVER="lightning", STEPS=3))
    assert state.step == 3
    assert (tmp_path / "smoke" / "export" / "smoke-comfy.safetensors").exists()


def test_sample_and_validation_callbacks(tmp_path):
    state = run(
        smoke_knobs(
            tmp_path,
            STEPS=2,
            SAMPLE_EVERY=2,
            SAMPLE_PROMPTS=["tag001, tag002", "tag003"],
            SAMPLE_SIZE=(128, 128),
            SAMPLE_STEPS=3,
            SAMPLE_SKIP_LAYERS=(1,),
            VALIDATE_EVERY=1,
            VALIDATION={
                "datasets": [{"source": "synthetic", "count": 4, "seed": 9}],
                "batches": 1,
            },
        )
    )
    assert state.step == 2
    folder = tmp_path / "smoke" / "samples" / "step_00000002"
    assert len(list(folder.glob("*.png"))) == 2
    settings = json.loads((folder / "settings.json").read_text())
    assert settings["steps"] == 3 and settings["prompts"] == [
        "tag001, tag002",
        "tag003",
    ]
    lines = [
        json.loads(x)
        for x in (tmp_path / "smoke" / "metrics.jsonl").read_text().splitlines()
    ]
    assert all("val_loss" in x for x in lines)


@pytest.mark.parametrize("logger", ["board", "wandb"])
def test_loggers(tmp_path, logger, monkeypatch):
    package = {"board": "kohakuboard", "wandb": "wandb"}[logger]
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} not installed")
    monkeypatch.setenv("WANDB_MODE", "offline")
    state = run(smoke_knobs(tmp_path, LOGGER=logger, STEPS=2))
    assert state.step == 2
    sub = {"board": "board", "wandb": "wandb"}[logger]
    assert (tmp_path / "smoke" / sub).exists()


def test_doctor_json(tmp_path, monkeypatch):
    out = tmp_path / "doctor.json"
    monkeypatch.setattr(
        sys, "argv", ["kohakumoss-doctor", "--json", str(out), "--memory"]
    )
    doctor.main()
    rep = json.loads(out.read_text())
    assert {"os", "python", "packages", "devices", "kernels", "memory_plans"} <= set(
        rep
    )
    assert all("rms_norm_mod" in t["float16"] for t in rep["kernels"].values())
    assert "cpu" in rep["kernels"]
