"""Fakes for pipeline tests: deterministic text encoder and VAE stand-ins."""

import hashlib

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.model import random_backbone


class FakeTextEncoder:
    """Prompt -> deterministic [len(words) + 2, text_dim] embedding."""

    def __init__(self, dim: int):
        self.dim = dim
        self.calls = 0

    def encode(self, prompts):
        self.calls += 1
        out = []
        for p in prompts:
            seed = int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed)
            out.append(torch.randn(len(p.split()) + 2, self.dim, generator=g))
        return out

    def empty(self):
        return self.encode([""])[0]


class FakeVae:
    def to(self, device):
        return self

    def decode(self, z):
        return torch.tanh(F.interpolate(z[:, :3].float(), scale_factor=16))


@pytest.fixture
def tiny64():
    return random_backbone("tiny", seed=11).double()


@pytest.fixture
def fake_te():
    return FakeTextEncoder(64)
