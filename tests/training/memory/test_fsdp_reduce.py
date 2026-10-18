"""FSDP_REDUCE: 'step' syncs FSDP2 gradients on the last micro-batch only, 'micro' on
every one; the memory estimate counts the unsharded gradients 'step' holds over
GRAD_ACC > 1 micro-batches (model-free: a recording root, the analytic estimate)."""

import importlib

import pytest

from kohakumoss.memory.profiles import resolve
from kohakumoss.model import resolve_config
from kohakumoss.training import knobs
from kohakumoss.training.build.model import fsdp_grads_sharded
from kohakumoss.training.build.parallel import fsdp_accumulate
from kohakumoss.training.knobs import validate
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

E = importlib.import_module("kohakumoss.memory.estimate")
pytestmark = pytest.mark.cpu


class Root:
    """Records `set_requires_gradient_sync` calls."""

    def __init__(self):
        self.calls: list[bool] = []

    def set_requires_gradient_sync(self, sync: bool) -> None:
        self.calls.append(sync)


def _syncs(reduce: str) -> list[bool]:
    root = Root()
    accumulate = fsdp_accumulate(root, reduce)
    for last in (False, False, True):
        with accumulate(last):
            pass
    return root.calls


def test_sync_modes():
    assert _syncs("step") == [False, False, True]
    assert _syncs("micro") == []
    must_fail(lambda: _equal(_syncs("micro"), [False, False, True]), "step for micro")
    with pytest.raises(ValueError):
        fsdp_accumulate(Root(), "bogus")


def _equal(a, b) -> None:
    assert a == b, (a, b)


def _grads(k: dict) -> int:
    arch = resolve_config("base")
    plan = resolve("ckpt-resident", {}, "cpu", trainable_base=True)
    work = E.Workload(
        arch,
        img_tokens=4096,
        txt_tokens=300,
        full_ft=True,
        world=4,
        sharded=True,
        sharded_grads=fsdp_grads_sharded(k),
    )
    return E.estimate(work, plan).parts["grads"]


def test_estimate_follows_the_knob():
    held = _grads({"GRAD_ACC": 2, "FSDP_REDUCE": "step"})
    sharded = _grads({"GRAD_ACC": 2, "FSDP_REDUCE": "micro"})
    one = _grads({"GRAD_ACC": 1, "FSDP_REDUCE": "step"})
    n = E.total_params(resolve_config("base"))
    # full-FT gradients are fp32, the width of the trainable masters
    assert held == n * 4 and sharded == n // 4 * 4 and one == sharded
    must_fail(lambda: _equal(sharded, held), "unsharded gradients not counted")


def _rejects_bogus() -> None:
    bad = [p for p in validate(runnable(FSDP_REDUCE="bogus")) if "FSDP_REDUCE" in p]
    assert bad, "an unknown FSDP_REDUCE passed"


def test_knob_is_validated(monkeypatch):
    _rejects_bogus()
    good = [p for p in validate(runnable(FSDP_REDUCE="micro")) if "FSDP_REDUCE" in p]
    assert good == []
    choices = {k: v for k, v in knobs.CHOICES.items() if k != "FSDP_REDUCE"}
    monkeypatch.setattr(knobs, "CHOICES", choices)
    must_fail(_rejects_bogus, "FSDP_REDUCE unchecked")
