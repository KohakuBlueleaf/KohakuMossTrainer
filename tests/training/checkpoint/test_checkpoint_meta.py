"""A checkpoint's trainable file carries the trainable set's metadata (what export and
inference read); the metadata reader matches safetensors' own."""

import pytest
import torch
from safetensors import safe_open

from kohakumoss.adapters.train.adapter_set import FOLD_KEY, RAW_TAG
from kohakumoss.training.checkpoint.io import save_checkpoint
from kohakumoss_common.io import read_safetensors_metadata
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
META = {RAW_TAG: "1", "kohakumoss_spec": '{"algo": "lora"}', FOLD_KEY: "null"}


def _saved(tmp_path, meta) -> str:
    trainable = {"lora_unet_x.lora_up.weight": torch.ones(2, 3)}
    out = save_checkpoint(tmp_path, trainable, {}, {}, {"step": 1}, meta)
    return str(out / "trainable.safetensors")


def _tagged(path: str) -> None:
    assert read_safetensors_metadata(path) == META


def test_the_trainable_file_records_its_metadata(tmp_path):
    path = _saved(tmp_path / "a", META)
    _tagged(path)
    with safe_open(path, framework="pt") as f:
        assert f.metadata() == META
    must_fail(lambda: _tagged(_saved(tmp_path / "b", None)), "no metadata written")
    assert read_safetensors_metadata(_saved(tmp_path / "c", None)) == {}
