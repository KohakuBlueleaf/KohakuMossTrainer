"""The offline text cache: the encoder-id guard, and the real-weights factory."""

import os
from pathlib import Path

import pytest
import torch

from kohakumoss_data import DataConfig
from kohakumoss_data.cache.build import build_text
from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.scripting import build_encoder
from tests._util.encode_fns import ByteCodes
from tests._util.negative import must_fail


def _dataset(tmp_path, encoder_id: str, spec: str = "") -> MossDataset:
    return MossDataset(
        DataConfig(
            datasets=[{"source": "synthetic", "count": 4}],
            batching="count",
            batch_size=2,
            target_area=128 * 128,
            text_cache="offline",
            latent_cache="off",
            cache_dir=str(tmp_path / "cache"),
            caption_variants=1,
            text_encoder_id=encoder_id,
            text_encoder_spec=spec,
        )
    )


@pytest.mark.cpu
def test_stand_in_encoder_cannot_write_under_a_real_id(tmp_path):
    def check(encoder_id):
        try:
            build_text(_dataset(tmp_path / encoder_id, encoder_id), ByteCodes(dim=8))
        except ValueError:
            return
        raise AssertionError(f"wrote stand-in embeddings under {encoder_id!r}")

    check("qwen3.5-9b")
    must_fail(lambda: check(ByteCodes.encoder_id), "matching stand-in id refused")


@pytest.mark.real_weights
@pytest.mark.cuda
def test_factory_builds_the_real_cache(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    root = Path(os.environ["KOHAKUMOSS_WEIGHTS"])
    path = next(p for p in (root / "base", root) if (p / "text_encoder").is_dir())
    encode = build_encoder(
        "kohakumoss.encoders.text.text_encoder",
        {"path": str(path / "text_encoder"), "plan": {"memory": "stream"}},
    )
    assert encode.encoder_id == "qwen3.5-9b"
    with pytest.raises(ValueError, match="TEXT_ENCODER_SPEC"):
        build_text(_dataset(tmp_path / "other", encode.encoder_id, "fp16/int8"), encode)
    ds = _dataset(tmp_path, encode.encoder_id, encode.encoder_spec)
    n = build_text(ds, encode, batch_size=64)
    assert n > 0
    out = encode(["a cat"])[0]
    assert out.device.type == "cpu" and out.shape[-1] == 4096
    cached = ds.texts.get(ds.texts.key(""))
    assert cached is not None and cached.shape[-1] == 4096
