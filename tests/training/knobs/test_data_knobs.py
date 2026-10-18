"""Every DataConfig field the docs describe is a train-script knob and reaches the data
config."""

import dataclasses

import pytest

from kohakumoss.training.knobs import KNOWN, validate
from kohakumoss_data import DataConfig
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
# latent_factor: fixed by the MOSS VAE + patchify; the encoder specs: derived by the
# trainer from its encoder knobs (training.build.model.data_knobs).
INTERNAL = {"latent_factor", "latent_encoder_spec", "text_encoder_spec"}


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
