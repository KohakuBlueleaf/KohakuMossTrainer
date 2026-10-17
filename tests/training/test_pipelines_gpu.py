"""PARALLEL threaded / pipeline through the entry on two CUDA devices (Linux)."""

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.training.entry import run
from kohakumoss.types.arch import PRESETS
from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.cache.random_encoders import RandomTextEncoder
from tests.training.test_entry_smoke import smoke_knobs

pytestmark = [pytest.mark.cuda, pytest.mark.multi_gpu, pytest.mark.slow]
EAGER = {
    op: "eager"
    for op in (
        "rms_norm",
        "rms_norm_mod",
        "qk_norm_rope",
        "attention_varlen",
        "situ_glu",
        "gated_residual",
        "token_mse",
    )
}


def _data_knobs(tmp_path, **extra) -> dict:
    """Static-shape data (token batches of <= 2 samples, exact budget from an offline
    text cache), shared by the single-device reference and the pipelined run."""
    knobs = {
        "KERNELS": EAGER,
        "BATCHING": "tokens",
        "MAX_TOKENS": 1024,
        "MAX_SAMPLES": 2,
        "TOKEN_BUDGET": "exact",
        "TEXT_CACHE": "offline",
        "CACHE_DIR": str(tmp_path / "cache"),
        "MICROBATCHES": 2,
        **extra,
    }
    probe = smoke_knobs(tmp_path, **{**knobs, "TOKEN_BUDGET": "estimate"})
    build_text(
        MossDataset(DataConfig.from_knobs(probe)),
        RandomTextEncoder(dim=PRESETS["tiny"].text_dim),
    )
    return knobs


def _trained(tmp_path, name, data: dict, **kw) -> dict:
    knobs = smoke_knobs(tmp_path / name, STEPS=2, SAVE_EVERY=2, **data, **kw)
    run(knobs)  # spawned kinds return no state; the step-2 checkpoint witnesses it
    ck = tmp_path / name / "smoke" / "checkpoints" / "step_00000002"
    return load_file(str(ck / "trainable.safetensors"))


TOKEN_DROP = {"MAX_TOKENS": 240, "OVERLONG": "drop_tokens"}  # 256-token images


@pytest.mark.parametrize(
    "kind, extra",
    [("threaded", {}), ("pipeline", {}), ("pipeline", TOKEN_DROP)],
    ids=["threaded", "pipeline", "pipeline-token-drop"],
)
def test_pipelined_matches_single_device(tmp_path, kind, extra):
    data = _data_knobs(tmp_path, **extra)
    ref = _trained(tmp_path, "ref", data, GRAD_ACC=2)
    schedule = {"PIPELINE_SCHEDULE": "gpipe"} if kind == "threaded" else {}
    got = _trained(tmp_path, kind, data, PARALLEL=kind, DEVICES=(0, 1), **schedule)
    assert ref.keys() == got.keys()
    changed = 0
    for key, value in ref.items():
        assert torch.allclose(value, got[key], rtol=1e-3, atol=1e-5), key
        changed += int(not torch.equal(value, torch.zeros_like(value)))
    assert changed > 0
