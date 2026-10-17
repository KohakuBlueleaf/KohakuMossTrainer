"""Memory profiles, validation, the memory model and the auto planner."""

import pytest

from kohakumoss.training.memory.estimate import (
    Workload,
    block_params,
    estimate,
    saved_per_token,
    total_params,
)
from kohakumoss.training.memory.planner import auto_plan
from kohakumoss.training.memory.profiles import PROFILES, resolve, validate
from kohakumoss.types.arch import PRESETS
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
BASE = PRESETS["base"]


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_profile_resolves(name):
    plan = resolve(name, device="cpu")
    assert plan.profile == name
    assert validate(plan) == []


def test_quant_resolves_per_arch_and_overrides_apply():
    plan = resolve("low", {"checkpoint": "selective"}, device="cpu")
    assert plan.base_precision == "int8"
    assert plan.checkpoint == "selective"


def test_invalid_combinations_listed_together():
    with pytest.raises(ValueError) as err:
        resolve("balanced", {"stream": "ring", "checkpoint": "none", "compile": "x"})
    msg = str(err.value)
    assert "checkpointed" in msg and "compile" in msg


def test_unknown_override_rejected():
    with pytest.raises(ValueError):
        resolve("balanced", {"nope": 1})


def test_full_ft_maps_ring_to_swap_and_keeps_dense_base():
    plan = resolve("minimal", device="cpu", trainable_base=True)
    assert plan.stream == "swap" and plan.base_precision == "compute"


def test_inference_drops_activation_knobs_and_allows_streaming():
    plan = resolve("minimal", device="cpu", inference=True)
    assert plan.checkpoint == "none" and not plan.act_offload
    assert plan.stream == "ring"
    assert validate(plan, inference=True) == []
    assert validate(plan) != []


def test_param_count_matches_release():
    assert total_params(BASE) == 5_082_308_608
    assert block_params(BASE) == 113_252_608


def test_estimate_orders_profiles():
    work = Workload(BASE, img_tokens=4096, txt_tokens=300, trainable_params=74_000_000)
    totals = [
        estimate(work, resolve(n)).total
        for n in ("speed", "balanced", "low", "minimal")
    ]
    assert totals == sorted(totals, reverse=True)
    assert saved_per_token(BASE, "none") == 12 * 3072 + 3 * 8192


def test_auto_picks_fastest_fitting_and_uses_probe():
    work = Workload(BASE, img_tokens=4096, txt_tokens=300, trainable_params=74_000_000)
    est = {n: estimate(work, resolve(n)).total for n in PROFILES}
    budget = (est["balanced"] + est["speed"]) // 2
    plan, report = auto_plan(work, "cpu", budget=budget)
    assert plan.profile == "balanced" and report.chosen == "balanced"

    def probe(p: MemoryPlan) -> int:
        return budget + 1

    plan, report = auto_plan(work, "cpu", budget=budget, probe=probe)
    assert plan.profile != "balanced"

    def wrong_order_check():
        p, _ = auto_plan(work, "cpu", budget=est["speed"] * 2)
        assert p.profile == "balanced"

    must_fail(wrong_order_check, "auto must prefer speed when it fits")
