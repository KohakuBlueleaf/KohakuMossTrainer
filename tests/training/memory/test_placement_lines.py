"""Training logs where the text encoder and the VAE were placed (the inference
placement lines), or why it holds none (model-free: placement records over a
one-parameter holder)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kohakumoss.training.build.model import placement_lines
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
K = {
    "WEIGHTS": "w",
    "VAE": "",
    "TEXT_ENCODER": "",
    "SAMPLE_EVERY": 0,
}


class _Holder(nn.Module):
    """Owns one parameter: the bytes the VAE line sizes."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, 8))


def _built(te=None, vae=None):
    return SimpleNamespace(text_encoder=te, vae=vae, feed=None)


def _vae(name: str):
    return SimpleNamespace(
        vae=_Holder(), placement=SimpleNamespace(name=name), device="cuda:0"
    )


def _messages(built) -> list[str]:
    return [m for m, _ in placement_lines(K, built, "cuda:0")]


def check_lines(lines: list[str]) -> None:
    assert lines[0].startswith("text encoder: not kept"), lines
    assert "VAE: " in lines[1] and "ring (host RAM" in lines[1], lines


def test_lines_name_each_component():
    check_lines(_messages(_built(vae=_vae("ring"))))
    none = _messages(_built())
    assert none[1].startswith("VAE: not loaded"), none
    must_fail(lambda: check_lines(none), "no VAE read as streamed")
