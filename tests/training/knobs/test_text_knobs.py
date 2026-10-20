"""TEXT_ENCODE / TEXT_DEVICE validation: the host runs the encoder with every text
cache mode."""

import pytest

from kohakumoss.training.checks.encoders import text_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _knobs(**kw):
    return {"TEXT_ENCODE": {}, "TEXT_DEVICE": "", "TEXT_CACHE": "offline", **kw}


def assert_host_accepted(problems_of) -> None:
    for cache in ("offline", "online", "off"):
        assert problems_of(_knobs(TEXT_DEVICE="cpu", TEXT_CACHE=cache)) == []
        k = _knobs(TEXT_ENCODE={"memory": "cpu"}, TEXT_CACHE=cache)
        assert problems_of(k) == []


def _refusing(k):
    if k["TEXT_CACHE"] != "offline":
        return ["the text encoder runs on the CPU only for the offline cache pass"]
    return text_problems(k)


def test_cpu_text_encoder_with_every_cache():
    assert_host_accepted(text_problems)
    must_fail(lambda: assert_host_accepted(_refusing), "the host refused online")


def test_plan_fields_and_choices():
    def flagged(k, needle):
        assert any(needle in m for m in text_problems(k)), text_problems(k)

    flagged(_knobs(TEXT_ENCODE={"memroy": "ring"}), "unknown text-encoder plan")
    flagged(_knobs(TEXT_ENCODE={"layout": "packed", "forward": "hf"}), "forward='hf'")
    flagged(_knobs(TEXT_ENCODE={"dtype": "fp32"}), "dtype='fp32'")
    for memory in ("ring", "stream"):
        must_fail(
            lambda: flagged(_knobs(TEXT_ENCODE={"memory": memory}), "text-encoder"),
            "a valid plan flagged",
        )
