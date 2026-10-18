"""TEXT_ENCODE / TEXT_DEVICE validation: the host only for the offline cache pass."""

import pytest

from kohakumoss.training.checks.encoders import text_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _knobs(**kw):
    return {"TEXT_ENCODE": {}, "TEXT_DEVICE": "", "TEXT_CACHE": "offline", **kw}


def test_cpu_text_encoder_only_with_offline_cache():
    def rejected(k):
        assert any("CPU only for the offline cache" in m for m in text_problems(k))

    rejected(_knobs(TEXT_DEVICE="cpu", TEXT_CACHE="online"))
    rejected(_knobs(TEXT_ENCODE={"memory": "cpu"}, TEXT_CACHE="off"))
    assert text_problems(_knobs(TEXT_DEVICE="cpu")) == []
    must_fail(lambda: rejected(_knobs(TEXT_DEVICE="cpu")), "offline cache refused")


def test_plan_fields_and_choices():
    def flagged(k, needle):
        assert any(needle in m for m in text_problems(k)), text_problems(k)

    flagged(_knobs(TEXT_ENCODE={"memroy": "stream"}), "unknown text-encoder plan")
    flagged(_knobs(TEXT_ENCODE={"layout": "packed", "forward": "hf"}), "forward='hf'")
    flagged(_knobs(TEXT_ENCODE={"dtype": "fp32"}), "dtype='fp32'")
    must_fail(
        lambda: flagged(_knobs(TEXT_ENCODE={"memory": "stream"}), "text-encoder"),
        "a valid plan flagged",
    )
