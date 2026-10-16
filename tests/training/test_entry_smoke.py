"""End-to-end `run(globals())` on the random tiny model, synthetic data, CPU."""

import json

import pytest

from kohakumoss.training.entry import run
from tests.training.test_knobs import runnable

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def smoke_knobs(tmp_path, **kw) -> dict:
    base = dict(
        DATASETS=[{"source": "synthetic", "count": 16}],
        LATENT_CACHE="off",
        TEXT_CACHE="off",
        RESOLUTION_MODE="bucket",
        TARGET_AREA=256 * 256,
        BATCH_SIZE=2,
        NUM_WORKERS=0,
        STEPS=3,
        SCHEDULE={"mode": "cosine", "warmup": 1, "end": 8},
        LOG_EVERY=1,
        SAVE_EVERY=2,
        DIM=4,
        ALPHA=4.0,
        OUTPUT_DIR=str(tmp_path),
        RUN_NAME="smoke",
        EXPORT_FORMATS=("comfy", "lycoris"),
    )
    return runnable(**{**base, **kw})


def test_lora_smoke_and_resume(tmp_path):
    state = run(smoke_knobs(tmp_path))
    assert state.step == 3
    out = tmp_path / "smoke"
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(x)["loss"] > 0 for x in lines)
    assert (out / "export" / "smoke-comfy.safetensors").exists()
    assert (out / "checkpoints" / "step_00000003" / "state.json").exists()
    state = run(smoke_knobs(tmp_path, STEPS=5, RESUME="latest"))
    assert state.step == 5


def test_fused_step_with_accumulation(tmp_path):
    state = run(
        smoke_knobs(tmp_path, MEMORY_OVERRIDES={"fused_step": True}, GRAD_ACC=2)
    )
    assert state.step == 3 and state.skipped == 0


def test_full_finetune_smoke(tmp_path):
    state = run(
        smoke_knobs(
            tmp_path,
            ALGO="full",
            EXPORT_FORMATS=("merged",),
            OPTIMIZER="muon",
            EMA=0.99,
        )
    )
    assert state.step == 3
    assert (tmp_path / "smoke" / "export" / "smoke-merged.safetensors").exists()
