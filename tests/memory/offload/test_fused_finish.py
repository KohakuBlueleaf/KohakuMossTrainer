"""In-backward updates on side streams are finished before the bundle's flag read and
the next forward: a value read on the compute stream right after `step()` is the
updated one even when the side stream is slow. The compute stream is a created stream:
the legacy default stream's implicit synchronisation would hide a missing wait."""

import sys

import pytest
import torch

from kohakumoss.offload.grad import GradOffload
from kohakumoss.training.optim.build import build_bundle
from kohakumoss.types.plans import PrecisionPolicy
from tests._util.negative import must_fail

pytestmark = pytest.mark.cuda
SLEEP = 200_000_000
HYPER = {"lr": 1e-1, "betas": (0.9, 0.999), "eps": 1e-8}


def _bundle(name, params):
    groups = [{"params": params, "lr": 1e-1, "weight_decay": 0.0}]
    policy = PrecisionPolicy(compute=torch.float32, scaler=False)
    return build_bundle(
        groups,
        params[0].device,
        name,
        HYPER,
        {"mode": "constant"},
        10,
        policy,
        0.0,
        fused_step=True,
    )


def _slow_side_stream(opt):
    update = opt._update

    def slow(p, ahead, found):
        with torch.cuda.stream(opt.opt):
            torch.cuda._sleep(SLEEP)
        update(p, ahead, found)

    opt._update = slow


def _slow_current_stream(bundle):
    update = bundle.update

    def slow(p, found):
        torch.cuda._sleep(SLEEP)
        update(p, found)

    bundle.update = slow


def _host_state(finish: bool):
    """One warm step (no allocation, so no cudaMalloc sync, in the measured one)."""
    p = torch.nn.Parameter(torch.ones(1 << 16, device="cuda"))
    bundle = _bundle("host_state_adamw", [p])
    for warm in (True, False):
        if not warm:
            _slow_side_stream(bundle.shared)
            if not finish:
                bundle.finishers = []
        p.grad = torch.ones_like(p)
        bundle.step_param(p)
        if warm:
            bundle.step()
            torch.cuda.synchronize()
            before = p.detach().clone()
    return p, bundle, before


def _grad_offload(finish: bool):
    p = torch.nn.Parameter(torch.ones(1 << 16, device="cuda"))
    bundle = _bundle("master_adamw", [p])
    grads = GradOffload([p])
    bundle.attach(grads, finishes=True)
    for warm in (True, False):
        if not warm:
            _slow_current_stream(bundle)
            if not finish:
                bundle.finishers = []
        bundle.arm(True)
        (p * 1.0).sum().backward()
        if warm:
            bundle.step()
            torch.cuda.synchronize()
            before = p.detach().clone()
    return p, bundle, before


WDDM = pytest.mark.skipif(
    sys.platform == "win32",
    reason="WDDM: the flag read's D2H copy queues behind the state write-back copy",
)


@pytest.mark.parametrize(
    "make",
    [pytest.param(_host_state, marks=WDDM), _grad_offload],
    ids=["host_state", "grad_offload"],
)
def test_step_waits_for_side_stream_updates(make):
    def check(finish: bool):
        with torch.cuda.stream(torch.cuda.Stream()):
            p, bundle, before = make(finish)
            bundle.step()
            seen = p.detach().clone()
        torch.cuda.synchronize()
        assert torch.equal(seen, p.detach()) and not torch.equal(seen, before)

    check(True)
    must_fail(lambda: check(False), "no finish before the compute stream reads")
