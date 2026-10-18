"""On this card, the encoder compute specs the trainer's cache keys record
(`data_knobs`) are the specs the real encoders report: the streamed text encoder of the
train-check settings and the cache script's VAE encoder. Offline caches the cache
scripts write on the dataset slice are hit in full under their recorded specs, and
under another text precision / VAE dtype every entry misses before any encoder loads,
with the miss error naming the spec knob."""

import runpy
from pathlib import Path

import pytest
import torch
from kohakuengine import run as run_script

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.encoders.vae import vae_encoder
from kohakumoss.training.build.model import data_knobs, training_text_plan
from kohakumoss_data.cache.build import caption_set, latent_items
from kohakumoss_data.cache.settings import offline_miss, recorded
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import data_config
from tests._util.dataset import kohya_source
from tests._util.negative import must_fail
from tests.integration.conftest import real_knobs

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "data"
KINDS = {  # cache kind -> (script, spec knob, DataConfig spec field, cache mode field)
    "text": ("cache_text.py", "TEXT_ENCODER_SPEC", "text_encoder_spec", "text_cache"),
    "latents": (
        "cache_latents.py",
        "LATENT_ENCODER_SPEC",
        "latent_encoder_spec",
        "latent_cache",
    ),
}


def check_specs(k_data: dict, text_spec: str, latent_spec: str) -> None:
    assert k_data["TEXT_ENCODER_SPEC"] == text_spec, (
        k_data["TEXT_ENCODER_SPEC"],
        text_spec,
    )
    assert k_data["LATENT_ENCODER_SPEC"] == latent_spec, (
        k_data["LATENT_ENCODER_SPEC"],
        latent_spec,
    )


def test_cache_specs_match_the_encoders(weights_root, dataset_slice, tmp_path):
    device = torch.device("cuda")
    k = real_knobs(weights_root, dataset_slice, tmp_path)
    k_data = data_knobs(k, device)
    plan, te_device = training_text_plan(k, device)
    te = QwenTextEncoder.from_pretrained(weights_root / "base", te_device, plan)
    text_spec = te.encoder_spec
    del te
    vae = vae_encoder(weights_root / "base", device, k.get("VAE_DTYPE", "auto"))
    latent_spec = vae.encoder_spec
    del vae
    torch.cuda.empty_cache()
    check_specs(k_data, text_spec, latent_spec)
    other = "fp32" if "fp32" not in latent_spec else "bf16"
    off = data_knobs({**k, "VAE_DTYPE": other}, device)
    must_fail(lambda: check_specs(off, text_spec, latent_spec), "another VAE dtype")


def _defaults(script: str) -> dict:
    """The script's UPPER_CASE knob defaults (its module globals, main not run)."""
    found = runpy.run_path(str(SCRIPTS / script), run_name="knob_defaults")
    return {k: v for k, v in found.items() if k.isupper()}


@pytest.fixture(scope="module")
def built_cache(weights_root, dataset_slice, tmp_path_factory) -> dict:
    """The data knobs of offline caches written by the two cache scripts on the
    dataset slice (VAE in its auto dtype, the text encoder streamed)."""
    data = {
        "DATASETS": [kohya_source(dataset_slice)],
        "CACHE_DIR": str(tmp_path_factory.mktemp("offline_cache")),
        "TARGET_AREA": 512 * 512,
    }
    encoders = {
        "latents": {
            "IMAGE_ENCODER_KWARGS": {
                "path": str(weights_root / "base" / "vae"),
                "device": "cuda",
            }
        },
        "text": {
            "TEXT_ENCODER_KWARGS": {
                "path": str(weights_root / "base" / "text_encoder"),
                "device": "cuda",
                "plan": {"memory": "stream"},
            }
        },
    }
    for kind, (script, *_) in KINDS.items():
        sets = {**data, **encoders[kind]}
        run_script(str(SCRIPTS / script), set_overrides=sets)
    torch.cuda.empty_cache()
    return data


def missing(kind: str, knobs: dict, spec: str) -> tuple[int, int, object]:
    """(missing entries, entries, DataConfig) of the offline `kind` cache under `spec`;
    counted from the keys alone, no encoder built."""
    script, _, field, mode = KINDS[kind]
    caches = {"text_cache": "off", "latent_cache": "off", mode: "offline"}
    cfg = data_config({**_defaults(script), **knobs}, **caches, **{field: spec})
    ds = MossDataset(cfg)
    if kind == "text":
        keys = [ds.texts.key(c) for c in caption_set(ds)]
        store = ds.texts.store
    else:
        keys = [ds.latents.key(ds.catalog[i.row_id], i) for i in latent_items(ds)]
        store = ds.latents.store
    return sum(key not in store for key in keys), len(keys), cfg


def check_hits(kind: str, knobs: dict, spec: str) -> None:
    n, total, _ = missing(kind, knobs, spec)
    assert total > 0 and n == 0, (kind, n, total, spec)


def _other(spec: str) -> str:
    """A spec of another dtype than `spec` (its first field)."""
    head, sep, rest = spec.partition("/")
    return ("fp32" if head != "fp32" else "bf16") + sep + rest


@pytest.mark.parametrize("kind", list(KINDS))
def test_offline_cache_refuses_another_encoder_spec(kind, built_cache):
    _, knob, field, _ = KINDS[kind]
    builds = recorded(built_cache["CACHE_DIR"], kind)
    assert builds and builds[-1].get(field), builds
    spec = builds[-1][field]
    check_hits(kind, built_cache, spec)
    other = _other(spec)
    must_fail(lambda: check_hits(kind, built_cache, other), f"{knob}={other!r}")
    n, total, cfg = missing(kind, built_cache, other)
    assert n == total
    message = str(offline_miss(cfg, kind, n, "first"))
    assert knob in message, message
