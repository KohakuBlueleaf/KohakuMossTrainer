"""Dataset-entry knob checks: a caption template that follows the row's centre crop
(`gbc`) refuses a random crop and a flip; a source's own checks are reported."""

import pytest

from kohakumoss_data import DataConfig
from kohakumoss_data.config import entry_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

GBC = {"template": "gbc", "formats": [{"weight": 1, "sections": ["nl"]}]}


def _cfg(tmp_path, **kw) -> DataConfig:
    entry = {"source": "folder", "path": str(tmp_path), "caption": GBC}
    return DataConfig(datasets=[entry], **kw)


def check(cfg, refused: bool) -> None:
    hit = any("follows the row's centre crop" in p for p in entry_problems(cfg))
    assert hit == refused


def test_geometric_caption_needs_centre_crop_and_no_flip(tmp_path):
    check(_cfg(tmp_path), False)
    check(_cfg(tmp_path, crop="random"), True)
    check(_cfg(tmp_path, flip=True), True)
    check(_cfg(tmp_path, resolution_mode="shifted_square"), True)
    must_fail(lambda: check(_cfg(tmp_path, flip=True), False), "flip accepted")


def test_gbc_source_without_a_dataset_is_reported(tmp_path):
    cfg = DataConfig(datasets=[{"source": "gbc", "path": str(tmp_path)}])
    assert any("gbc" in p for p in entry_problems(cfg))
