"""Build-time specialisation: each config builds the bound per-step callable it needs,
and the callback schedule fires exactly where a per-step modulo check would."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.diffusion.objective import HuberLoss, L1Loss
from kohakumoss.diffusion.objective.loss import _masked_sum, _plain_sum
from kohakumoss.diffusion.timestep import LogitNormalTimestep, MossTimestep
from kohakumoss.diffusion.timestep.simple import _identity
from kohakumoss.model import random_backbone
from kohakumoss.training import step as step_mod
from kohakumoss.training.build.loop import build_batcher, build_step
from kohakumoss.training.callbacks import CallbackRunner
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.checkpoint.manager import _no_write
from kohakumoss.training.drivers.common import last_flags
from kohakumoss.training.ema import ConstantEMA
from kohakumoss.training.ema.averagers import _fixed_decay, _warm_decay
from kohakumoss.training.optim import build_bundle
from kohakumoss.training.optim.bundle import _clip_off
from kohakumoss.training.precision import DynamicScaler, NoScaler, build_policy
from kohakumoss.training.step import OnlineText
from tests._util.negative import must_fail
from tests.training.test_knobs import runnable

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")


def _built():
    model = random_backbone("tiny", device=DEV, dtype=torch.float32, seed=0)
    enc = SimpleNamespace(encode=lambda prompts: [], encode_images=lambda i, hw: None)
    return SimpleNamespace(backbone=model, text_encoder=enc, vae=enc)


def test_step_and_batcher_bind_mask_and_encoders():
    built = _built()
    plain = build_step(runnable(), built, None, None, DEV)
    assert (
        plain.mask_of is step_mod._no_mask and plain.count_of is step_mod._image_count
    )
    masked = build_step(runnable(LOSS="masked_mse"), built, None, None, DEV)
    assert masked.mask_of is step_mod._batch_mask
    assert masked.count_of is step_mod._mask_count
    cached = build_batcher(runnable(), built, DEV)
    assert (
        cached.text is step_mod._cached_text and cached.mask is step_mod._no_mask_move
    )
    online = build_batcher(runnable(TEXT_CACHE="off", LOSS="masked_mse"), built, DEV)
    assert isinstance(online.text, OnlineText) and online.mask is step_mod._mask_move


def test_losses_samplers_ema_bind_once():
    assert HuberLoss(None).reduce is _plain_sum
    assert L1Loss(None, masked=True).reduce is _masked_sum
    assert MossTimestep().sample.__func__ is MossTimestep._single
    assert MossTimestep(schedule="tech_report").sample.__func__ is MossTimestep._staged
    assert LogitNormalTimestep(shift=1.0).shift is _identity
    assert LogitNormalTimestep(shift=3.0).shift is not _identity
    p = torch.nn.Parameter(torch.zeros(2))
    assert ConstantEMA([("p", p)], (0.9,), warmup=False).rules[0].func is _fixed_decay
    assert ConstantEMA([("p", p)], (0.9,), warmup=True).rules[0].func is _warm_decay


def test_bundle_scaler_and_writer_selection(tmp_path):
    model = torch.nn.Linear(4, 4)
    groups = [{"params": list(model.parameters()), "lr": 1e-3, "weight_decay": 0.0}]
    hyper = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8}
    sched = {"mode": "constant", "end": -1}
    bf16 = build_bundle(
        groups, DEV, "adamw", hyper, sched, 10, build_policy("bf16", ()), 0.0
    )
    assert isinstance(bf16.scaler, NoScaler) and bf16.clip is _clip_off
    fp16 = build_bundle(
        groups, DEV, "adamw", hyper, sched, 10, build_policy("fp16", ()), 1.0
    )
    assert isinstance(fp16.scaler, DynamicScaler) and fp16.clip is not _clip_off
    fused = build_bundle(
        groups,
        DEV,
        "adamw",
        hyper,
        sched,
        10,
        build_policy("fp16", ()),
        1.0,
        fused_step=True,
    )
    assert fused.clip_param.func.__name__ == "_clip_param"
    assert CheckpointManager(tmp_path, 1, is_main=False)._save is _no_write
    assert last_flags(3) == (False, False, True) and last_flags(1) == (True,)


class _Rec:
    def __init__(self, every, log):
        self.every = every
        self.log = log

    def on_step_end(self, state, out):
        self.log.append((state.step, self.every))


def _fires(runner, steps, log):
    for s in range(1, steps + 1):
        runner.run(SimpleNamespace(step=s), None)
    return log


def _reference(everies, steps):
    return [(s, e) for s in range(1, steps + 1) for e in everies if s % e == 0]


def test_callback_schedule_matches_modulo():
    everies = (1, 3, 5, 10)
    log = []
    runner = CallbackRunner([_Rec(e, log) for e in everies])
    assert [cb.every for cb in runner.always] == [1]
    assert _fires(runner, 60, log) == _reference(everies, 60)
    log2 = []
    resumed = CallbackRunner([_Rec(e, log2) for e in everies])
    resumed.reset(7)
    for s in range(8, 31):
        resumed.run(SimpleNamespace(step=s), None)
    assert log2 == [x for x in _reference(everies, 30) if x[0] >= 8]

    def broken():
        log3 = []
        bad = CallbackRunner([_Rec(e, log3) for e in everies])
        bad.periods = []
        bad.reset(0)
        assert _fires(bad, 60, log3) == _reference(everies, 60)

    must_fail(broken, "periodic callbacks never scheduled")
