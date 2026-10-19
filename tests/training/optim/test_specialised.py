"""Build-time specialisation: each config builds the bound per-step callable it needs,
and the callback schedule fires exactly where a per-step modulo check would."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.diffusion.objective import HuberLoss, L1Loss
from kohakumoss.diffusion.objective.loss import _masked_sum, _plain_sum
from kohakumoss.diffusion.timestep import LogitNormalTimestep, MossTimestep
from kohakumoss.precision import DynamicScaler, NoScaler, build_precision
from kohakumoss.training.callbacks import CallbackRunner
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers.common import last_flags
from kohakumoss.training.ema import ConstantEMA
from kohakumoss.training.ema.averagers import _fixed_decay, _warm_decay
from kohakumoss.training.optim import build_bundle
from kohakumoss_common.calls import identity, nothing
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")


def test_losses_samplers_ema_bind_once():
    assert HuberLoss(None).reduce is _plain_sum
    assert L1Loss(None, masked=True).reduce is _masked_sum
    assert MossTimestep()._sample is MossTimestep._single
    assert MossTimestep(schedule="tech_report")._sample is MossTimestep._staged
    assert LogitNormalTimestep(shift=1.0).shift is identity
    assert LogitNormalTimestep(shift=3.0).shift is not identity
    p = torch.nn.Parameter(torch.zeros(2))
    assert ConstantEMA([("p", p)], (0.9,), warmup=False).rules[0].func is _fixed_decay
    assert ConstantEMA([("p", p)], (0.9,), warmup=True).rules[0].func is _warm_decay


def test_bundle_scaler_and_writer_selection(tmp_path):
    params = [torch.nn.Parameter(torch.zeros(4, 4)), torch.nn.Parameter(torch.zeros(4))]
    groups = [{"params": params, "lr": 1e-3, "weight_decay": 0.0}]
    hyper = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8}
    sched = {"mode": "constant", "end": -1}
    bf16 = build_bundle(
        groups, DEV, "adamw", hyper, sched, 10, build_precision("bf16", ()), 0.0
    )
    assert isinstance(bf16.scaler, NoScaler) and bf16.clip is nothing
    fp16 = build_bundle(
        groups, DEV, "adamw", hyper, sched, 10, build_precision("fp16", ()), 1.0
    )
    assert isinstance(fp16.scaler, DynamicScaler) and fp16.clip is not nothing
    fused = build_bundle(
        groups,
        DEV,
        "adamw",
        hyper,
        sched,
        10,
        build_precision("fp16", ()),
        1.0,
        fused_step=True,
    )
    assert fused.factor is type(fused)._clipped
    unclipped = build_bundle(
        groups, DEV, "adamw", hyper, sched, 10, build_precision("fp16", ()), 0.0
    )
    assert unclipped.clip is nothing
    assert CheckpointManager(tmp_path, 1, is_main=False)._save is nothing
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
