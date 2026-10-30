"""STEP_SYNC='deferred': DeviceScaler follows DynamicScaler's scale sequence from a
device flag, the gated decay-to-one, the sync cadence and knob problems (CPU); and on
CUDA a DeferredBundle with torch's fused AdamW steps exactly like the per-step Bundle
through an overflowing step (parameters, skip count, scale)."""

import pytest
import torch

from kohakumoss.precision.scaler import DeviceScaler, DynamicScaler
from kohakumoss.training.optim.bundle.base import Bundle
from kohakumoss.training.optim.bundle.deferred import (
    DeferredBundle,
    gated_decay_to_one,
    step_sync_problems,
    sync_cadence,
)
from tests._util.negative import must_fail

pytestmark = [pytest.mark.budget(20)]

FLAGS = [0, 0, 1, 0, 0, 0, 1, 1, 1, 1] + [0] * 10


def _scales(flags, update) -> list[float]:
    host = DynamicScaler(init=8.0, growth_interval=3, floor=2.0)
    dev = DeviceScaler(DynamicScaler(init=8.0, growth_interval=3, floor=2.0), "cpu")
    a, b = [], []
    for f in flags:
        host.update(bool(f))
        update(dev, torch.tensor(float(f)))
        a.append(host.scale)
        b.append(dev.host_scale())
    return a, b


@pytest.mark.cpu
def test_device_scaler_matches_host():
    a, b = _scales(FLAGS, DeviceScaler.update_device)
    assert a == b
    assert min(a) == 2.0 and max(a) > 8.0


@pytest.mark.cpu
def test_device_scaler_negative():
    def no_floor(self, bad):
        clean = (self.clean + 1.0) * (1.0 - bad)
        grow = (clean >= self.growth_interval).float()
        grown = self.scale * (1.0 + (self.growth - 1.0) * grow)
        self.scale = torch.where(bad > 0, self.scale * self.backoff, grown)
        self.clean = clean * (1.0 - grow)

    def check():
        a, b = _scales(FLAGS, no_floor)
        assert a == b

    must_fail(check, "backoff without the floor")


@pytest.mark.cpu
def test_device_scaler_state_round_trip():
    dev = DeviceScaler(DynamicScaler(init=8.0, growth_interval=3), "cpu")
    dev.update_device(torch.tensor(0.0))
    state = dev.state_dict()
    other = DeviceScaler(DynamicScaler(), "cpu")
    other.load_state_dict(state)
    assert other.state_dict() == state == {"scale": 8.0, "clean_steps": 1}


@pytest.mark.cpu
def test_gated_decay_to_one():
    p = torch.nn.Parameter(torch.full((3,), 3.0))
    opt = torch.optim.SGD([p], lr=0.5)
    apply = gated_decay_to_one(
        [{"params": [p], "decay_to_one": True, "decay_to_one_coef": 0.1}], opt
    )
    apply(torch.tensor(1.0))
    assert torch.equal(p.detach(), torch.full((3,), 3.0))
    apply(torch.tensor(0.0))
    assert torch.allclose(p.detach(), torch.full((3,), 3.0 - 0.05 * 2.0))
    assert gated_decay_to_one([{"params": [p]}], opt) is None


@pytest.mark.cpu
def test_cadence_and_problems():
    assert sync_cadence(10, 10) == 10 and sync_cadence(10, 4) == 2
    assert step_sync_problems({"STEP_SYNC": "step", "EMA": 0.999}) == []
    assert step_sync_problems({"STEP_SYNC": "deferred", "EMA": 0.0}) == []
    assert step_sync_problems({"STEP_SYNC": "deferred", "EMA": 0.999})
    assert step_sync_problems({"STEP_SYNC": "deferred", "PARALLEL": "fsdp"})


def _bundle(kind, **kw):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64, device="cuda"))
    groups = [{"params": [p], "lr": 1e-2, "weight_decay": 0.01}]
    opt = torch.optim.AdamW(groups, lr=1e-2, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    scaler = DynamicScaler(init=1024.0, growth_interval=2, floor=1.0)
    bundle = kind(opt, sched, scaler, groups=groups, grad_clip=1.0, **kw)
    return bundle, p


def _run(kind, **kw):
    bundle, p = _bundle(kind, **kw)
    x = torch.linspace(-1, 1, 64, device="cuda")
    for step in range(1, 7):
        loss = ((p * x).sum() - 1.0).square()
        if step == 3:
            loss = loss * float("inf")
        bundle.backward(loss)
        bundle.step()
        if hasattr(bundle, "sync_due"):
            bundle.sync_due(step)
    if hasattr(bundle, "sync"):
        bundle.sync()
    scale = bundle.host_scale if hasattr(bundle, "host_scale") else bundle.scaler.scale
    return p.detach().clone(), bundle.skipped, scale


@pytest.mark.cuda
def test_deferred_matches_step():
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    pa, skipped_a, scale_a = _run(Bundle)
    pb, skipped_b, scale_b = _run(DeferredBundle, every=2, last=6)
    assert skipped_a == skipped_b == 1
    assert scale_a == scale_b
    assert torch.equal(pa, pb)
