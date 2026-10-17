"""End-to-end `run(globals())` on the random tiny model, synthetic data, CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.training.entry import deterministic_mode, run
from tests._util.negative import must_fail
from tests.training.test_knobs import runnable

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]


def kogine(*argv: str) -> subprocess.CompletedProcess:
    """`kogine <argv>` in a child process from the repo root, on CPU."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    cmd = [sys.executable, "-c", "from kohakuengine.cli import main; main()", *argv]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def _determinism() -> tuple:
    return (
        torch.are_deterministic_algorithms_enabled(),
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        torch.backends.cudnn.benchmark,
    )


def test_default_training_is_not_deterministic_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("KOHAKUMOSS_DETERMINISTIC", raising=False)
    before = _determinism()
    assert before[0] is False
    run(smoke_knobs(tmp_path, STEPS=1, SAVE_EVERY=0))
    assert _determinism() == before

    monkeypatch.setenv("KOHAKUMOSS_DETERMINISTIC", "1")
    try:
        deterministic_mode()
        on = _determinism()
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = before[2]
        if before[1] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    assert on[0] is True and on[1] == ":4096:8"
    must_fail(lambda: _same_state(on, before), "the switch does nothing")


def _same_state(a, b) -> None:
    assert a == b


def test_kogine_runs_train_script(tmp_path):
    out = kogine(
        "run",
        "scripts/train/moss.py",
        "--config",
        "configs/moss/smoke/smoke_tiny.py",
        "--set",
        f"OUTPUT_DIR={tmp_path.as_posix()}",
        "--set",
        "STEPS=3",
        "--set",
        "SAVE_EVERY=3",
    )
    assert out.returncode == 0, out.stderr[-4000:]
    lines = (tmp_path / "tiny" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(x)["loss"] > 0 for x in lines)
    assert (tmp_path / "tiny" / "export" / "tiny-comfy.safetensors").exists()


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


DETERMINISTIC = os.environ.get("KOHAKUMOSS_DETERMINISTIC") == "1"


def exact_runs() -> dict:
    """Bit-exact run comparisons: always on CPU; on CUDA only in deterministic mode
    (test_deterministic_resume.py runs these tests in a child process with it on),
    with LyCORIS on its torch tier: its fused tiers have no deterministic backward yet
    (reported to the LyCORIS owner). Returns the knobs the compared runs share."""
    if not torch.cuda.is_available():
        return {}
    if not DETERMINISTIC:
        pytest.skip("CUDA run equality runs under KOHAKUMOSS_DETERMINISTIC=1")
    return {"KERNELS": {"lycoris": "torch"}}


def trainable_at(out: Path, step: int) -> dict:
    ck = out / "smoke" / "checkpoints" / f"step_{step:08d}"
    return load_file(str(ck / "trainable.safetensors"))


def assert_same(a: dict, b: dict, rtol: float = 0.0, atol: float = 0.0) -> None:
    assert a.keys() == b.keys()
    for key in a:
        torch.testing.assert_close(
            a[key], b[key], rtol=rtol, atol=atol, msg=lambda m, k=key: f"{k}: {m}"
        )


def assert_trained(state: dict) -> None:
    ups = [v for k, v in state.items() if "lora_up" in k]
    assert ups and max(float(v.abs().max()) for v in ups) > 0, "nothing trained"


def test_lora_smoke_and_resume(tmp_path):
    same = exact_runs()
    full = tmp_path / "full"
    run(smoke_knobs(full, STEPS=5, SAVE_EVERY=0, **same))
    state = run(smoke_knobs(tmp_path, **same))
    assert state.step == 3
    out = tmp_path / "smoke"
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(x)["loss"] > 0 for x in lines)
    assert (out / "export" / "smoke-comfy.safetensors").exists()
    assert (out / "checkpoints" / "step_00000003" / "state.json").exists()
    state = run(smoke_knobs(tmp_path, STEPS=5, RESUME="latest", **same))
    assert state.step == 5
    resumed, straight = trainable_at(tmp_path, 5), trainable_at(full, 5)
    assert_trained(resumed)
    assert_same(resumed, straight)
    must_fail(
        lambda: assert_same(trainable_at(tmp_path, 3), straight),
        "resume compared against a different step",
    )


def test_fused_step_with_accumulation(tmp_path):
    fused = tmp_path / "fused"
    plain = tmp_path / "plain"
    kw = dict(GRAD_ACC=2, GRAD_CLIP=0.0, SAVE_EVERY=3)
    state = run(smoke_knobs(fused, MEMORY_OVERRIDES={"fused_step": True}, **kw))
    assert state.step == 3 and state.skipped == 0 and state.nonfinite == 0
    run(smoke_knobs(plain, **kw))
    got = trainable_at(fused, 3)
    assert_trained(got)
    assert_same(got, trainable_at(plain, 3), rtol=1e-4, atol=1e-7)
    zero = {k: v * 0 if "lora_up" in k else v for k, v in got.items()}
    must_fail(lambda: assert_trained(zero), "fused path trained nothing")


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
