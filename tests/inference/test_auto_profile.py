"""MEMORY_PROFILE='auto' in sampling: the fastest profile (ORDER) whose inference
estimate fits the budget, checked at each profile's exact estimate, and an error
naming the estimates when none fits; the sample knobs accept 'auto'."""

import pytest

from kohakumoss.inference import knobs
from kohakumoss.inference.loading import auto_profile
from kohakumoss.inference.placement_report import REFERENCE_TOKENS, forward_tokens
from kohakumoss.memory.estimate import Workload, estimate_inference
from kohakumoss.memory.profiles import ORDER, resolve
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ARCH = PRESETS["base"]
WORK = Workload(arch=ARCH, img_tokens=REFERENCE_TOKENS, txt_tokens=0, compute_bytes=2)


def needs() -> dict[str, int]:
    """Each profile's inference estimate of the reference forward, bytes."""
    return {
        name: estimate_inference(WORK, resolve(name, None, "cpu", inference=True)).total
        for name in ORDER
    }


def check(choose) -> None:
    """`choose(budget)` picks, at each profile's own estimate, that profile unless a
    faster one fits too; below every estimate it raises naming them."""
    need = needs()
    for i, name in enumerate(ORDER):
        expected = next(n for n in ORDER[: i + 1] if need[n] <= need[name])
        assert choose(need[name]) == expected, (name, need)
    try:
        choose(min(need.values()) - 1)
    except ValueError as err:
        assert "no profile fits" in str(err), err
    else:
        raise AssertionError("a budget below every estimate was accepted")


def test_auto_picks_the_fastest_profile_that_fits():
    check(lambda budget: auto_profile(ARCH, 2, "cpu", None, budget))
    need = needs()

    def slowest(budget: int) -> str:
        return next(n for n in reversed(ORDER) if need[n] <= budget)

    must_fail(lambda: check(slowest), "the slowest fitting profile")
    assert "auto" in knobs.CHOICES["MEMORY_PROFILE"]


def need_at(name: str, tokens: int) -> int:
    work = Workload(arch=ARCH, img_tokens=tokens, txt_tokens=0, compute_bytes=2)
    plan = resolve(name, None, "cpu", inference=True)
    return estimate_inference(work, plan).total


def _fits(choose, tokens: int, budget: int) -> None:
    """The chosen profile's estimate at `tokens` fits `budget`."""
    name = choose(budget)
    assert need_at(name, tokens) <= budget, (name, need_at(name, tokens), budget)


def test_auto_sizes_for_the_call_largest_forward():
    assert REFERENCE_TOKENS == forward_tokens([(1024, 1024)], 1, 3.0, "packed")
    four = forward_tokens([(1024, 1024), (832, 1216)], 4, 3.0, "packed")
    assert four == 2 * 4 * (64 * 64 + 512)
    assert forward_tokens([(1024, 1024)], 4, 1.0, "packed") == four // 2
    budget = need_at("speed", REFERENCE_TOKENS)
    _fits(lambda b: auto_profile(ARCH, 2, "cpu", None, b, four), four, budget)

    def reference(b: int) -> str:
        return auto_profile(ARCH, 2, "cpu", None, b, REFERENCE_TOKENS)

    must_fail(lambda: _fits(reference, four, budget), "sized for one sample")
