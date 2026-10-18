"""The tokenizer is found next to the text-encoder model."""

import pytest

from kohakumoss.encoders.text import tokenizer_dir
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_tokenizer_found_from_root_or_model_dir(tmp_path):
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "tokenizer").mkdir()
    assert tokenizer_dir(tmp_path) == tmp_path / "tokenizer"
    assert tokenizer_dir(tmp_path / "text_encoder") == tmp_path / "tokenizer"
    alone = tmp_path / "other" / "alone"
    alone.mkdir(parents=True)
    assert tokenizer_dir(alone) == alone


def test_model_dir_with_tokenizer_files_wins_over_sibling(tmp_path):
    model = tmp_path / "text_encoder"
    model.mkdir()
    (tmp_path / "tokenizer").mkdir()
    (model / "tokenizer_config.json").write_text("{}")

    def check(found):
        assert found == model

    check(tokenizer_dir(model))
    (model / "tokenizer_config.json").unlink()
    must_fail(lambda: check(tokenizer_dir(model)), "sibling tokenizer/ preferred")
