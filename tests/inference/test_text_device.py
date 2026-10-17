"""The text encoder runs on its own device; embeddings land on the compute device."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kohakumoss.encoders.text import QwenTextEncoder, tokenizer_dir
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

DIM = 8


class FakeTokenizer:
    def apply_chat_template(self, messages, **kw):
        return messages[0]["content"]

    def __call__(self, text, **kw):
        return {"input_ids": torch.arange(len(text) + 1).view(1, -1)}


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, DIM)

    def forward(self, input_ids, **kw):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


def check_placement(te: QwenTextEncoder, run_on: str, land_on: str) -> None:
    out = te.encode(["a cat", "a dog"]) + [te.empty()]
    assert all(t.device.type == land_on for t in out), [t.device for t in out]
    assert next(te.model.parameters()).device.type == run_on
    assert [t.shape[0] for t in out] == [6, 6, 1]


def test_cpu_encoder_lands_on_output_device():
    te = QwenTextEncoder(FakeModel(), FakeTokenizer(), "cpu", output_device="meta")
    check_placement(te, "cpu", "meta")
    stay = QwenTextEncoder(FakeModel(), FakeTokenizer(), "cpu")
    must_fail(lambda: check_placement(stay, "cpu", "meta"), "outputs left on CPU")


def test_tokenizer_found_from_root_or_model_dir(tmp_path):
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "tokenizer").mkdir()
    assert tokenizer_dir(tmp_path) == tmp_path / "tokenizer"
    assert tokenizer_dir(tmp_path / "text_encoder") == tmp_path / "tokenizer"
    alone = tmp_path / "other" / "alone"
    alone.mkdir(parents=True)
    assert tokenizer_dir(alone) == alone


@pytest.mark.cuda
def test_cpu_encoder_feeds_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    te = QwenTextEncoder(FakeModel(), FakeTokenizer(), "cpu", output_device="cuda")
    check_placement(te, "cpu", "cuda")
