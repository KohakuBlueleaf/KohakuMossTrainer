"""Component roots: a named variant must exist; the resolved roots land in settings."""

from pathlib import Path

import pytest
import torch

from kohakumoss.inference import MossPipeline
from kohakumoss.inference import pipeline as pipeline_module
from kohakumoss.model import random_backbone, save_diffusers
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _silent_root(path: Path, variant, name):
    for root in ((path / variant) if variant else None, path, path / "base"):
        if root is not None and (root / name).is_dir():
            return root
    raise FileNotFoundError(name)


def _expect_missing(resolve, path):
    try:
        resolve(path, "sft", "transformer")
    except FileNotFoundError:
        return
    raise AssertionError("missing variant resolved")


@pytest.fixture
def weights(tmp_path):
    model = random_backbone("tiny", seed=3)
    save_diffusers(model, tmp_path / "base" / "transformer", dtype=torch.float32)
    return tmp_path


def test_missing_variant_raises(weights):
    _expect_missing(pipeline_module._component_root, weights)
    must_fail(lambda: _expect_missing(_silent_root, weights), "silent fallback")


def test_settings_record_sources(weights):
    kw = dict(dtype=torch.float32, device="cpu", tune="off")
    kw.update(text_encoder=False, vae=False)
    pipe = MossPipeline.from_pretrained(weights, variant="base", **kw)
    embeds = [torch.randn(4, pipe.backbone.cfg.text_dim)]
    res = pipe.generate(
        prompt_embeds=embeds, sizes=(32, 32), steps=1, cfg=1.0, output="latent"
    )
    assert res.settings["sources"] == {"transformer": str(weights / "base")}
    with pytest.raises(FileNotFoundError):
        MossPipeline.from_pretrained(weights, variant="sft", **kw)
