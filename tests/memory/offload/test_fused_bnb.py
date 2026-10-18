"""Fused step with 8-bit states (bitsandbytes): each in-backward parameter update
equals the stock optimiser's step and issues no device synchronize."""

from functools import partial

import pytest
import torch

from kohakumoss.training.optim.bundle import fused
from kohakumoss.training.optim.bundle.build import build_bundle
from kohakumoss.types.plans import PrecisionConfig
from tests._util.negative import must_fail

bnb = pytest.importorskip("bitsandbytes")
bnb_optimizer = pytest.importorskip("bitsandbytes.optim.optimizer")
pytestmark = pytest.mark.cuda
HYPER = {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8}
SHAPES = [(256, 64), (64,)]


def _params():
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(s, device="cuda")) for s in SHAPES]


def _grads(step: int) -> list[torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(step)
    return [torch.randn(s, device="cuda", generator=g) for s in SHAPES]


def _fused_run(syncs: list):
    params = _params()
    groups = [{"params": params, "lr": 1e-2, "weight_decay": 0.0}]
    prec = PrecisionConfig(compute=torch.float32, scaler=False)
    bundle = build_bundle(
        groups,
        params[0].device,
        "adamw8bit",
        HYPER,
        {"mode": "constant"},
        10,
        prec,
        0.0,
        fused_step=True,
    )
    for step in range(3):
        for p, g in zip(params, _grads(step)):
            p.grad = g
            before = len(syncs)
            bundle.step_param(p)
            assert len(syncs) == before, "device synchronize inside step_param"
        bundle.step()
    return params


def _stock_run():
    params = _params()
    opt = bnb.optim.AdamW8bit(params, weight_decay=0.0, **HYPER)
    for step in range(3):
        for p, g in zip(params, _grads(step)):
            p.grad = g
        opt.step()
    return params


def test_fused_8bit_matches_stock_without_syncs(monkeypatch):
    syncs = []
    real = bnb_optimizer.sync_gpu
    monkeypatch.setattr(
        bnb_optimizer, "sync_gpu", lambda t: (syncs.append(1), real(t))[1]
    )
    got = _fused_run(syncs)
    want = _stock_run()
    for a, b in zip(got, want):
        assert torch.allclose(a, b, rtol=0, atol=1e-6)
    stock = partial(fused._step_kept, fused._step_plain)
    monkeypatch.setattr(fused, "_step_runner", lambda opt: stock)
    must_fail(lambda: _fused_run(syncs), "the stock step syncs per parameter")
