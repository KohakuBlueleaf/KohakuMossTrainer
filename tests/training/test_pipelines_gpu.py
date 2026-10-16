"""PARALLEL threaded / pipeline through the entry on two CUDA devices (Linux)."""

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.training.entry import run
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


def _trained(tmp_path, name, **kw) -> dict:
    knobs = smoke_knobs(
        tmp_path / name,
        STEPS=2,
        SAVE_EVERY=2,
        KERNELS=EAGER,
        BATCHING="count",
        BATCH_SIZE=1,
        MAX_TOKENS=1024,
        MICROBATCHES=2,
        **kw,
    )
    run(knobs)  # spawned kinds return no state; the step-2 checkpoint witnesses it
    ck = tmp_path / name / "smoke" / "checkpoints" / "step_00000002"
    return load_file(str(ck / "trainable.safetensors"))


@pytest.mark.parametrize("kind", ["threaded", "pipeline"])
def test_pipelined_matches_single_device(tmp_path, kind):
    ref = _trained(tmp_path, "ref", GRAD_ACC=2)
    got = _trained(tmp_path, kind, PARALLEL=kind, DEVICES=(0, 1))
    assert ref.keys() == got.keys()
    changed = 0
    for key, value in ref.items():
        assert torch.allclose(value, got[key], rtol=1e-3, atol=1e-5), key
        changed += int(not torch.equal(value, torch.zeros_like(value)))
    assert changed > 0
