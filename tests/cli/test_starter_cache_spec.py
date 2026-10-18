"""The starter's data config caches what its train config keys on: the latent and text
cache passes' encoder specs (from the data config's encoder kwargs, filled from the
train knobs VAE_DTYPE / TEXT_ENCODE) equal the trainer's `data_knobs` specs, on a bf16
and an fp16 card, with the default and an fp8 text-encoder precision."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.cli.config import starter
from kohakumoss.cli.config.validate import TRAIN, compose
from kohakumoss.encoders.specs import latent_encoder_kwargs, text_encoder_kwargs
from kohakumoss.encoders.text.factory import text_encoder_spec
from kohakumoss.training.build.model import data_knobs
from tests._util.encoder_specs import vae_encoder_spec
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]


def card(monkeypatch, bf16: bool) -> None:
    """A CUDA card with or without bf16, for the dtype resolution of both sides."""
    arch = SimpleNamespace(has_bf16=bf16, name="card")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("kohakumoss.encoders.vae.arch_of", lambda device: arch)
    monkeypatch.setattr("kohakumoss.encoders.text.plan.arch_of", lambda device: arch)


def pair(tmp_path: Path, text_encode: dict) -> tuple[dict, dict]:
    """(train knobs, data-config knobs) of the starter pair for a 24 GiB card."""
    c = starter.TrainChoice(
        weights=str(tmp_path / "w"), data=str(tmp_path / "d"), prompts=("a cat",)
    )
    st = starter.train_starter(ROOT, c)
    train = compose(ROOT, TRAIN, st.layers, st.own)
    train["TEXT_ENCODE"] = {**dict(train["TEXT_ENCODE"] or {}), **text_encode}
    data = {**train, **starter.data_starter(c, tmp_path / "train.py").own}
    return train, data


def specs(train: dict, data: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    """((trainer latent, text spec), (latent cache pass, text cache pass spec))."""
    k = data_knobs(train, torch.device("cuda"))
    trainer = (k["LATENT_ENCODER_SPEC"], k["TEXT_ENCODER_SPEC"])
    latent = vae_encoder_spec(**latent_encoder_kwargs(data))
    text = text_encoder_spec(**text_encoder_kwargs(data))
    return trainer, (latent, text)


def same(train: dict, data: dict) -> None:
    trainer, cache = specs(train, data)
    assert trainer == cache, (trainer, cache)


@pytest.mark.parametrize("bf16", [True, False])
@pytest.mark.parametrize("text_encode", [{}, {"precision": "fp8_compute"}])
def test_the_data_config_caches_what_the_run_keys_on(
    tmp_path, monkeypatch, bf16, text_encode
):
    card(monkeypatch, bf16)
    train, data = pair(tmp_path, text_encode)
    same(train, data)
    (latent, text), _ = specs(train, data)
    assert latent == ("bf16" if bf16 else "fp16")
    assert text.startswith(latent + "/" + text_encode.get("precision", "compute"))
    kwargs = {**data["IMAGE_ENCODER_KWARGS"], "dtype": "fp32"}
    fp32 = {**data, "IMAGE_ENCODER_KWARGS": kwargs}
    must_fail(lambda: same(train, fp32), "an fp32 VAE kwarg in the data config")
    if text_encode:
        unread = {**data, "TEXT_ENCODE": {}}
        must_fail(lambda: same(train, unread), "TEXT_ENCODE not read by the cache")
