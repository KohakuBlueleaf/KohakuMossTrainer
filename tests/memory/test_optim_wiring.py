"""The resolved optimiser wiring of a MemoryPlan, model-free (no streamed blocks):
OPTIMIZER follows optim_state, the in-backward step and DDP role come from the plan
and the layout hooks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from kohakumoss.memory.apply import MemoryEngines
from kohakumoss.training.memory.optimizers import resolve_wiring
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
DDP = SimpleNamespace(ddp=object())
CASES = (
    (MemoryPlan(), None, ("adamw", False, True, False, False)),
    (
        MemoryPlan(optim_state="cpu", fused_step=True),
        None,
        ("cpu_adamw", True, False, True, False),
    ),
    (
        MemoryPlan(optim_state="stream"),
        DDP,
        ("host_state_adamw", False, False, False, True),
    ),
)


def assert_wiring(resolve) -> None:
    for plan, hooks, (name, fused, resident, host_adamw, has_ddp) in CASES:
        w = resolve(plan, MemoryEngines(plan), "adamw", hooks)
        assert w.name == name
        assert w.fused is fused
        assert w.resident_states is resident
        assert w.host_adamw is host_adamw
        assert (w.ddp is not None) is has_ddp
        assert not w.gpu_blocks and not w.sharded and not w.streamed


def test_resolve_wiring():
    assert_wiring(resolve_wiring)


def test_wiring_check_fails_on_broken_resolve():
    def unmapped(plan, engines, optimizer, hooks):
        return replace(resolve_wiring(plan, engines, optimizer, hooks), name=optimizer)

    def unfused(plan, engines, optimizer, hooks):
        return replace(resolve_wiring(plan, engines, optimizer, hooks), fused=False)

    must_fail(lambda: assert_wiring(unmapped), "optimizer not following optim_state")
    must_fail(lambda: assert_wiring(unfused), "fused_step dropped")
