"""Every knob value, one factor at a time over the tiny smoke: a combination that
`validate` accepts must train (steps advance, finite loss, parameters move); one it
rejects must be rejected by `check` before anything is built."""

import importlib.util
import json

import pytest
import torch

from kohakumoss.training.entry import run
from kohakumoss.training.knobs import validate, with_precision
from tests._util.negative import must_fail
from tests.training.test_entry_smoke import smoke_knobs, trainable_at

pytestmark = [pytest.mark.cpu, pytest.mark.slow]

SYNTH = [{"source": "synthetic", "count": 16}]
SAMPLE = dict(
    SAMPLE_EVERY=2,
    SAMPLE_PROMPTS=["a cat"],
    SAMPLE_STEPS=2,
    SAMPLE_SIZE=(128, 128),
    SAMPLE_SKIP_LAYERS=(),
)
FACTORS = {
    "algo": [
        {"ALGO": a}
        for a in ("locon", "loha", "lokr", "oft", "boft", "glora", "ia3", "dylora")
    ]
    + [{"ALGO": "tlora"}],
    "full": [
        {"ALGO": "full", "EXPORT_FORMATS": ("merged", "diffusers")},
        {"ALGO": "full", "FULL_MODE": "lycoris", "EXPORT_FORMATS": ("lycoris",)},
        {"ALGO": "full", "OPTIMIZER": "muon", "EXPORT_FORMATS": ("merged",)},
        {"ALGO": "full", "TRAIN_TEXT_REFINER": True, "EXPORT_FORMATS": ("merged",)},
    ],
    "adapter": [
        {"DORA": True},
        {"ADAPTER_MODE": "merge"},
        {"TRAIN_NORMS": True},
        {"LORAPLUS_RATIO": 4.0},
        {"PRESET": "attn"},
        {"PRESET": "full-block"},
        {"DROPOUT": {"input": 0.1, "rank": 0.1, "module": 0.1}},
        {"EXPORT_FORMATS": ("merged", "diffusers", "lycoris", "comfy")},
    ],
    "objective": [
        {"TIMESTEP": "uniform", "TIMESTEP_KWARGS": {}},
        {"TIMESTEP": "logit_normal", "TIMESTEP_KWARGS": {}},
        {"TIMESTEP_KWARGS": {"schedule": "tech_report"}},
        {"LOSS": "huber"},
        {"LOSS": "l1"},
        {"WEIGHTING": "x0"},
        {"WEIGHTING": "eps"},
    ],
    "optim": [
        {"OPTIMIZER": "muon"},
        {"OPTIMIZER": "cpu_adamw"},
        {"OPTIMIZER": "prodigy", "LR": 1.0},
        {"OPTIMIZER": "schedule_free"},
        {"OPTIMIZER": "adamw8bit"},
        {"OPTIMIZER": "adamw16"},
        {"GRAD_CLIP": 0.0},
        {"WD_RULES": "none"},
        {"WD_RULES": "all"},
        {"SCHEDULE": {"mode": "constant", "end": -1}},
        {"EMA": 0.99},
        {"EMA": [0.9, 0.99]},
        {"EMA": {"kind": "power", "sigma_rel": [0.05, 0.1]}},
    ],
    "memory": [
        {"MEMORY_PROFILE": p} for p in ("speed", "balanced", "low", "minimal", "auto")
    ]
    + [
        {"MEMORY_OVERRIDES": {"fused_step": True}, "GRAD_ACC": 2},
        {"MEMORY_OVERRIDES": {"grad_offload": True}, "GRAD_ACC": 2},
        {"MEMORY_OVERRIDES": {"act_offload": True}},
        {"MEMORY_OVERRIDES": {"checkpoint": "none"}},
    ],
    "loop": [
        {"DRIVER": "lightning"},
        {"DRIVER": "lightning", "MEMORY_OVERRIDES": {"fused_step": True}},
        {"GRAD_ACC": 3},
        {"COMPILE": "block"},
        {"TUNE": "off"},
        {"SAVE_MINUTES": 0.0001, "SAVE_EVERY": 0},
        {"KEEP_LAST": 1, "SAVE_EVERY": 1},
        {**SAMPLE},
        {"VALIDATE_EVERY": 2, "VALIDATION": {"datasets": SYNTH, "batches": 2}},
    ],
    "data": [
        {"RESOLUTION_MODE": m} for m in ("resize", "shifted_square", "square_crop")
    ]
    + [
        {"BATCHING": "tokens", "MAX_TOKENS": 1024},
        {"BATCHING": "tokens_ffd", "MAX_TOKENS": 1024, "MAX_SAMPLES": 2},
        {"BATCHING": "tokens", "MAX_TOKENS": 300, "OVERLONG": "drop_tokens"},
        {"BATCHING": "tokens", "MAX_TOKENS": 300, "OVERLONG": "truncate_text"},
        {"TEXT_DROP": 0.5},
        {"FLIP": True},
        {"CAPTION_VARIANTS": 1},
        {"UPSCALE": "to_area"},
        {"CROP": "random"},
    ],
}
CASES = [
    (f"{group}-{i}", kw) for group, kws in FACTORS.items() for i, kw in enumerate(kws)
]
OPTIONAL = {"prodigy": "prodigyopt", "schedule_free": "schedulefree"}
OPTIONAL["adamw8bit"] = "bitsandbytes"


def _skip_missing(kw: dict) -> None:
    module = OPTIONAL.get(kw.get("OPTIMIZER", ""))
    if module and importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    if (
        kw.get("DRIVER") == "lightning"
        and importlib.util.find_spec("lightning") is None
    ):
        pytest.skip("lightning not installed")


def _trained(out, k: dict, before: dict) -> None:
    """Steps logged with finite losses, and the trainable state moved."""
    lines = (out / "smoke" / "metrics.jsonl").read_text().splitlines()
    losses = [json.loads(x)["loss"] for x in lines]
    assert len(losses) == k["STEPS"] and all(v == v and v < 1e4 for v in losses)
    after = trainable_at(out, k["STEPS"])
    assert after.keys() == before.keys() or not before
    moved = any(not torch.equal(after[n], before[n]) for n in before)
    assert moved or not before, "no trainable parameter changed"


@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_every_accepted_combination_trains(tmp_path, name, kw):
    _skip_missing(kw)
    k = smoke_knobs(tmp_path, **{"STEPS": 2, "SAVE_EVERY": 2, **kw})
    problems = validate(k)
    if problems:
        with pytest.raises(SystemExit):
            run(k)
        pytest.skip("rejected by validation: " + "; ".join(problems))
    base = smoke_knobs(tmp_path / "init", **{**kw, "STEPS": 1, "LR": 1e-30})
    state = run(k)
    assert state.step == 2
    k = with_precision(k)
    assert torch.isfinite(torch.tensor(state.loss))
    _trained(tmp_path, k, _initial(tmp_path / "init", base))


def _initial(out, k: dict) -> dict:
    """The trainable state of the same build after one step at a negligible lr
    (reference for "parameters moved")."""
    k["SAVE_EVERY"] = 1
    run(k)
    return trainable_at(out, 1)


def test_the_moved_check_fails_on_an_untrained_run(tmp_path):
    k = smoke_knobs(tmp_path, STEPS=1, SAVE_EVERY=1, LR=1e-30)
    run(k)
    frozen = trainable_at(tmp_path, 1)
    must_fail(lambda: _trained(tmp_path, with_precision(k), frozen), "nothing moved")
