"""Every DataConfig field the docs describe is a train-script knob, reaches the data
config, and LATENT_DRAW also selects the online VAE's draw."""

import dataclasses

import pytest
import torch

from kohakumoss.training.build import model as model_build
from kohakumoss.training.knobs import KNOWN, validate
from kohakumoss_data import DataConfig
from tests._util.negative import must_fail
from tests.training.test_knobs import runnable

pytestmark = pytest.mark.cpu
INTERNAL = {"latent_factor"}  # fixed by the MOSS VAE + patchify, not a knob


def _all_fields_are_knobs(known) -> None:
    fields = {f.name for f in dataclasses.fields(DataConfig)} - INTERNAL
    missing = sorted(f.upper() for f in fields if f.upper() not in known)
    assert not missing, missing


def test_every_data_field_is_a_knob():
    _all_fields_are_knobs(KNOWN)
    must_fail(lambda: _all_fields_are_knobs(KNOWN - {"CROP"}), "CROP not a knob")


def test_knobs_reach_the_data_config():
    k = runnable(
        CROP="random",
        PLANNER_KWARGS={"window": 8},
        LATENT_DRAW="sample",
        MAX_TEXT_TOKENS=512,
        PIN_MEMORY=False,
    )
    assert validate(k) == []
    cfg = DataConfig.from_knobs(k)
    assert cfg.crop == "random" and cfg.planner_kwargs == {"window": 8}
    assert cfg.latent_draw == "sample" and cfg.max_text_tokens == 512
    assert cfg.pin_memory is False


def test_latent_draw_selects_the_vae_draw(monkeypatch, tmp_path):
    seen = {}

    def spy(path, device=None, mode="mode", **kw):
        seen["mode"] = mode
        return object()

    monkeypatch.setattr(model_build.FluxVae, "from_pretrained", staticmethod(spy))
    k = runnable(
        WEIGHTS=str(tmp_path),
        LATENT_CACHE="online",
        TEXT_CACHE="offline",
        SAMPLE_EVERY=0,
        LATENT_DRAW="sample",
    )
    model_build.build_encoders(k, torch.device("cpu"), None)
    assert seen["mode"] == "sample"
