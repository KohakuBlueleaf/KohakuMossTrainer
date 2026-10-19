"""fp8 / MXFP8 compute bases, model-free: the text-encoder and memory-plan rules per
arch and the sample knobs' mapping to the memory plan. Quantized storage, per-call
activation quantization and fp8 training on the real DiT / text encoder run in
tests/integration/test_train_real.py and test_infer_real.py; the fp8 GEMMs themselves in
tests/kernels/quant/test_fp8.py."""

import pytest

from kohakumoss.encoders.text import text_plan
from kohakumoss.encoders.text.plan import plan_problems
from kohakumoss.inference import knobs
from kohakumoss.memory import profiles
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.hardware.arch import _cuda_arch
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _arch(sm):
    return lambda device: _cuda_arch(sm // 10, sm % 10)


def _assert(cond):
    assert cond


@pytest.mark.parametrize("precision", ("fp8_compute", "mxfp8"))
def test_text_plan_rejects_archs_without_fp8(precision):
    plan = text_plan({"precision": precision})
    for sm in (70, 75, 86):
        problems = plan_problems(plan, "cuda", arch=_arch(sm))
        assert any("sm89" in p for p in problems), (sm, problems)
    for sm in (89, 120):
        assert plan_problems(plan, "cuda", arch=_arch(sm)) == []
    must_fail(
        lambda: _assert(any("sm89" in p for p in plan_problems(plan, "cuda", arch=_arch(120)))),  # fmt: skip
        "the rule fires on an fp8 arch",
    )
    bad = plan_problems(text_plan({"precision": precision, "quant_group": 64}))
    assert any("quant_group" in p for p in bad)
    assert plan_problems(text_plan({"fp8_recipe": "mx"}))


@pytest.mark.parametrize("precision", ("fp8_compute", "mxfp8"))
def test_memory_plan_rejects_archs_without_fp8(precision, monkeypatch):
    plan = MemoryPlan(base_precision=precision)
    for sm, ok in ((75, False), (86, False), (89, True), (120, True)):
        monkeypatch.setattr(profiles, "arch_of", _arch(sm))
        problems = profiles.validate(plan, inference=True, device="cuda")
        assert (problems == []) == ok, (sm, problems)
    assert profiles.validate(plan, inference=True, device="cpu")
    assert profiles.validate(plan, trainable_base=True) == []
    swap = MemoryPlan(base_precision=precision, stream="swap", stream_resident=2)
    assert any(
        "stream='off'" in p for p in profiles.validate(swap, trainable_base=True)
    )
    assert profiles.validate(MemoryPlan(base_precision="int8"), trainable_base=True)
    assert profiles.validate(MemoryPlan(fp8_recipe="mx"))
    assert profiles.validate(MemoryPlan(fp8_dgrad="fp16"))


def test_sample_knobs_map_to_the_memory_plan():
    k = {
        "MEMORY_PROFILE": "",
        "MEMORY_OVERRIDES": {},
        "BASE_PRECISION": "mxfp8",
        "FP8_RECIPE": "row",
    }
    memory, overrides = knobs.memory_of(k)
    assert memory == "speed"
    assert overrides == {"base_precision": "mxfp8", "fp8_recipe": "row"}
    assert knobs.memory_of({**k, "BASE_PRECISION": "auto"}) == (None, {})
    compute = knobs.memory_of({**k, "BASE_PRECISION": "compute"})
    assert compute == (None, {"base_precision": "compute"})
    k2 = {**k, "MEMORY_PROFILE": "low", "MEMORY_OVERRIDES": {"fp8_recipe": "block"}}
    assert knobs.memory_of(k2) == ("low", {"base_precision": "mxfp8", "fp8_recipe": "block"})  # fmt: skip
