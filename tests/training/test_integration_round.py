"""FP16_FIX in the build, gradient offload, T-LoRA masks, pipeline knob rules."""

import sys

import pytest
import torch
from safetensors.torch import load_file

from kohakumoss.adapters import tlora
from kohakumoss.model import random_backbone
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import validate
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs
from tests.training.test_knobs import runnable

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _params(path) -> dict:
    return load_file(str(path))


def test_fp16_fix_online_trains(tmp_path):
    state = run(
        smoke_knobs(tmp_path, FP16_FIX="online", FP16_FIX_KWARGS={"c": 1 / 16}, STEPS=3)
    )
    assert state.step == 3 and state.skipped == 0


def test_fp16_fix_weights_merged_export_is_unfolded(tmp_path):
    knobs = dict(
        ALGO="full",
        EXPORT_FORMATS=("merged",),
        LR=1e-12,
        STEPS=1,
        SAVE_EVERY=0,
        FP16_FIX="weights",
        FP16_FIX_KWARGS={"c": 1 / 8},
    )
    run(smoke_knobs(tmp_path, **knobs))
    got = _params(tmp_path / "smoke" / "export" / "smoke-merged.safetensors")
    device = "cuda" if torch.cuda.is_available() else "cpu"  # the entry's init device
    ref = random_backbone("tiny", device=device, dtype=torch.float32, seed=0)
    ref = {k: v.cpu() for k, v in ref.state_dict().items()}
    key = "video_embedder.weight"
    assert torch.allclose(got[key].float(), ref[key], rtol=1e-2, atol=1e-3)

    def broken():
        assert torch.allclose(got[key].float(), ref[key] / 8, rtol=1e-2, atol=1e-3)

    must_fail(broken, "folded weights exported")


def _final(tmp_path, name, **kw):
    run(smoke_knobs(tmp_path / name, STEPS=3, GRAD_ACC=2, SAVE_EVERY=3, **kw))
    ck = tmp_path / name / "smoke" / "checkpoints" / "step_00000003"
    return load_file(str(ck / "trainable.safetensors"))


def test_grad_offload_device_optimiser_matches(tmp_path):
    base = _final(tmp_path, "plain")
    off = _final(tmp_path, "off", MEMORY_OVERRIDES={"grad_offload": True})
    for key, value in base.items():
        assert torch.allclose(value, off[key], atol=1e-6), key


def test_grad_offload_host_and_fused(tmp_path):
    host = _final(
        tmp_path, "host", MEMORY_OVERRIDES={"grad_offload": True, "optim_state": "cpu"}
    )
    fused = _final(
        tmp_path, "fused", MEMORY_OVERRIDES={"grad_offload": True, "fused_step": True}
    )
    assert all(torch.isfinite(v).all() for v in host.values())
    assert all(torch.isfinite(v).all() for v in fused.values())


def test_tlora_mask_set_every_micro_batch(tmp_path, monkeypatch):
    calls = []
    real = tlora.set_token_mask

    def spy(sigma, seg, max_rank, **kw):
        calls.append((sigma.shape[0], seg.shape[0], max_rank))
        return real(sigma, seg, max_rank, **kw)

    monkeypatch.setattr(tlora, "set_token_mask", spy)
    state = run(
        smoke_knobs(
            tmp_path, ALGO="tlora", STEPS=2, GRAD_ACC=2, EXPORT_FORMATS=("lycoris",)
        )
    )
    assert state.step == 2 and len(calls) == 4 and all(c[2] == 4 for c in calls)


def test_pipeline_layouts_rejected_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    for kind in ("pipeline", "threaded"):
        problems = validate(runnable(PARALLEL=kind, DEVICES=(0, 1), MICROBATCHES=2))
        assert any("Linux-only" in p for p in problems), kind
