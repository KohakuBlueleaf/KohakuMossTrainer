"""The offline-miss rebuild sets the knobs that give the encoder spec (VAE_DTYPE,
TEXT_ENCODE's dtype / precision / quant_group / fp8_recipe), never the spec knobs the
cache passes overwrite: each text spec read back through the cache pass's kwargs gives
the same spec."""

import pytest

from kohakumoss.encoders.specs import text_encoder_kwargs, text_spec
from kohakumoss.encoders.text.factory import text_encoder_spec
from kohakumoss.encoders.text.plan import DTYPES, text_plan
from kohakumoss_data.cache import settings
from kohakumoss_data.cache.settings import config_lines, rebuild_command, spec_knobs
from kohakumoss_data.config import DataConfig
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
PLANS = (
    {"dtype": "bf16"},
    {"dtype": "fp16", "precision": "int8", "quant_group": 64},
    {"dtype": "bf16", "precision": "fp8", "quant_group": 0},
    {"dtype": "bf16", "precision": "fp8_compute", "fp8_recipe": "block"},
    {"dtype": "fp16", "precision": "mxfp8"},
)


def check_round_trip(knobs_of) -> None:
    """spec -> knobs_of -> the text cache pass's spec equals the spec."""
    for fields in PLANS:
        plan = text_plan({**fields, "memory": "cpu"})
        spec = text_spec(plan, DTYPES[plan.dtype])
        cfg = DataConfig(text_encoder_spec=spec)
        k = {
            "TEXT_ENCODER": "kohakumoss.encoders.text_encoder",
            "TEXT_ENCODER_KWARGS": {"path": "w", "device": "cpu", "plan": {}},
            **knobs_of(cfg),
        }
        assert text_encoder_spec(**text_encoder_kwargs(k)) == spec, (fields, spec)


def test_the_spec_knobs_give_the_spec():
    check_round_trip(lambda cfg: spec_knobs(cfg, "text"))
    assert spec_knobs(DataConfig(latent_encoder_spec="fp16"), "latents") == {
        "VAE_DTYPE": "fp16"
    }

    def no_group(cfg) -> dict:
        k = spec_knobs(cfg, "text")
        k["TEXT_ENCODE"].pop("quant_group", None)
        return k

    must_fail(lambda: check_round_trip(no_group), "quant_group dropped")


def _sets_the_spec_knobs(cfg: DataConfig) -> None:
    cmd = rebuild_command(cfg, "text", quote=str)
    lines = config_lines(cfg, "text")
    assert "TEXT_ENCODER_SPEC" not in cmd and "TEXT_ENCODE=" in cmd, cmd
    assert "'quant_group': 64" in cmd, cmd
    assert any(line.startswith("TEXT_ENCODE = ") for line in lines), lines
    latents = rebuild_command(
        DataConfig(latent_encoder_spec="bf16"), "latents", quote=str
    )
    assert "VAE_DTYPE=bf16" in latents and "LATENT_ENCODER_SPEC" not in latents


def test_the_rebuild_command_sets_the_encoder_knobs(monkeypatch):
    cfg = DataConfig(cache_dir="c", text_encoder_spec="fp16/int8/g64")
    _sets_the_spec_knobs(cfg)
    monkeypatch.setattr(settings, "_knob_values", _spec_as_is)
    must_fail(lambda: _sets_the_spec_knobs(cfg), "the spec knob printed as is")


def _spec_as_is(cfg: DataConfig, kind: str) -> dict:
    return {f.upper(): v for f, v in settings.settings_of(cfg, kind).items()}
