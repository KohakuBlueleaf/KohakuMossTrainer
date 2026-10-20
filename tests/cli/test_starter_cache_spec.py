"""The starter's data config caches what its train config keys on: the latent and text
cache passes' encoder specs (from the data config's encoder kwargs, filled from the
train knobs VAE_DTYPE / TEXT_ENCODE) equal the trainer's `data_knobs` specs, on a bf16
and an fp16 card, with the default and an fp8 text-encoder precision, and with either
side's encoders on the host."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.cli.config import starter
from kohakumoss.cli.config.validate import TRAIN, compose
from kohakumoss.encoders.specs import latent_encoder_kwargs, text_encoder_kwargs
from kohakumoss.encoders.text.encoder import resolve_text_plan
from kohakumoss.encoders.text.factory import text_encoder_spec
from kohakumoss.encoders.vae import vae_encoder_spec
from kohakumoss.training.build.model import data_knobs
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


def _on_host(kwargs: dict) -> dict:
    return {**kwargs, "plan": {**dict(kwargs.get("plan") or {}), "memory": "cpu"}}


def placed(train: dict, data: dict, train_host: bool, cache_host: bool):
    """(train, data) knobs with the text encoder and the VAE on the host in the run
    (TEXT_DEVICE, VAE_ENCODE) and / or in the cache passes (the encoder kwargs)."""
    if train_host:
        vae = {**dict(train.get("VAE_ENCODE") or {}), "memory": "cpu"}
        train = {**train, "TEXT_DEVICE": "cpu", "VAE_ENCODE": vae}
    if cache_host:
        data = {
            **data,
            "TEXT_ENCODER_KWARGS": _on_host(data["TEXT_ENCODER_KWARGS"]),
            "IMAGE_ENCODER_KWARGS": _on_host(data["IMAGE_ENCODER_KWARGS"]),
        }
    return train, data


def _placement_dtype(path, plan, device, run_device=None):
    return resolve_text_plan(path, plan, device)


@pytest.mark.parametrize("bf16", [True, False])
@pytest.mark.parametrize("train_host", [False, True])
@pytest.mark.parametrize("cache_host", [False, True])
def test_the_cache_keys_ignore_where_the_encoders_compute(
    tmp_path, monkeypatch, bf16, train_host, cache_host
):
    card(monkeypatch, bf16)
    train, data = placed(*pair(tmp_path, {}), train_host, cache_host)
    same(train, data)
    if not bf16 and train_host != cache_host:
        for module in ("encoders.text.factory", "training.build.model"):
            monkeypatch.setattr(
                f"kohakumoss.{module}.resolve_text_plan", _placement_dtype
            )
        must_fail(lambda: same(train, data), "auto resolved where the pass computes")
