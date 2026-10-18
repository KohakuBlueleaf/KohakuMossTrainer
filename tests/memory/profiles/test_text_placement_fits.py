"""The text encoder's placement is the fastest one that fits the budget left by each
candidate DiT plan, not tied to whether the DiT streams: next to `balanced` it is
resident on a 40 GB budget and streams on an 8 GB one, and every named profile on a
96 GB budget keeps it resident (model-free: explicit text-encoder sizes)."""

from functools import partial

import pytest

from kohakumoss.memory.catalog_table import lora_params
from kohakumoss.memory.estimate import GIB, Workload, estimate
from kohakumoss.memory.planner import with_text
from kohakumoss.memory.profiles import ORDER, resolve
from kohakumoss.offload.encoders.text_memory import TextSizes
from kohakumoss.training.build.model import encoder_memory
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ARCH = PRESETS["base"]
WORK = Workload(ARCH, 4096, 300, trainable_params=lora_params(ARCH, 32))
TE = TextSizes(
    total=int(14.78 * GIB),
    embedding=int(1.89 * GIB),
    layers=int(12.89 * GIB),
    ring=int(1.59 * GIB),
    hidden=4096,
    intermediate=12288,
    value_heads=32,
    head_k=128,
    head_v=128,
    elem=2,
)
TOKENS = (16384, 16384)
RULE = partial(encoder_memory, {}, "text", "auto")


def _placed(rule, name: str, budget_gib: float) -> str:
    plan = resolve(name, device="cpu")
    est = estimate(WORK, plan)
    _, memory = with_text(est, TE, int(budget_gib * GIB), TOKENS, rule(plan))
    return memory


def check_fits(rule) -> None:
    assert _placed(rule, "balanced", 40) == "resident"
    assert _placed(rule, "balanced", 8) == "stream"
    for name in ORDER:
        assert _placed(rule, name, 96) == "resident", name


def _tied_to_dit(plan) -> str:
    """The rule that streams the text encoder whenever the DiT streams."""
    return "stream" if plan.stream != "off" else "auto"


def test_text_encoder_takes_the_fastest_fit():
    check_fits(RULE)
    must_fail(lambda: check_fits(_tied_to_dit), "the TE tied to the DiT's streaming")
