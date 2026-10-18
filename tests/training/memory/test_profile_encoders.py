"""Encoder placement with TEXT_ENCODE / VAE_ENCODE `memory` unset: the text encoder is
`auto` under every plan (the fastest placement that fits, decided against the budget);
the VAE follows the resolved memory plan, not the profile's name (offload next to
streamed DiT blocks, resident otherwise); an explicit `memory` always wins (model-free:
plans and knob dicts)."""

import pytest
import torch

from kohakumoss.training.build import model as model_build
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
STREAMED = MemoryPlan(stream="ring", stream_resident=2)
RESIDENT = MemoryPlan(stream="off")


def _k(**kw) -> dict:
    return {"MEMORY_PROFILE": "auto", "TEXT_ENCODE": {}, "VAE_ENCODE": {}, **kw}


def _text(plan, **kw) -> str:
    k = {"TEXT_DEVICE": "", **_k(**kw)}
    return model_build.training_text_plan(k, torch.device("cuda"), plan)[0].memory


def check_plan_rule() -> None:
    assert _text(STREAMED) == "auto" and _text(RESIDENT) == "auto"
    assert model_build.encoder_memory(_k(), "vae", "resident", STREAMED) == "offload"
    assert model_build.encoder_memory(_k(), "vae", "resident", RESIDENT) == "resident"
    assert _text(STREAMED, TEXT_ENCODE={"memory": "resident"}) == "resident"
    own = _k(VAE_ENCODE={"memory": "offload"})
    assert model_build.encoder_memory(own, "vae", "resident", RESIDENT) == "offload"


def test_placement_follows_the_plan(monkeypatch):
    check_plan_rule()
    assert "streamed" in model_build.encoder_rule(_k(), "vae", STREAMED)
    assert "fits" in model_build.encoder_rule(_k(), "text", STREAMED)
    monkeypatch.setattr(model_build, "streamed_base", lambda plan: False)
    must_fail(check_plan_rule, "a streamed DiT left the encoders resident")


def test_profile_name_alone_does_not_decide():
    named = _k(MEMORY_PROFILE="minimal")
    assert model_build.encoder_memory(named, "vae", "resident", RESIDENT) == "resident"
