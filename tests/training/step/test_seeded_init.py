"""The adapter init draws from SEED whatever ran before it (tuning, prewarm, cache
state), and leaves the caller's generator where it was (model-free, CPU)."""

import contextlib

import pytest
import torch

from kohakumoss.training.build.model import seeded_rng
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


def _init_draw(seeded) -> torch.Tensor:
    with seeded(7, CPU):
        return torch.randn(8)


def check_seeded(seeded) -> None:
    torch.manual_seed(0)
    first = _init_draw(seeded)
    torch.manual_seed(123)
    torch.randn(1000)
    second = _init_draw(seeded)
    assert torch.equal(first, second), "the init depends on earlier draws"
    torch.manual_seed(5)
    _init_draw(seeded)
    after = torch.randn(4)
    torch.manual_seed(5)
    assert torch.equal(after, torch.randn(4)), "the caller's stream moved"


@contextlib.contextmanager
def _unseeded(seed: int, device):
    with torch.random.fork_rng(devices=[]):
        yield


@contextlib.contextmanager
def _unrestored(seed: int, device):
    torch.manual_seed(seed)
    yield


def test_adapter_init_is_seeded():
    check_seeded(seeded_rng)
    must_fail(lambda: check_seeded(_unseeded), "init without SEED")
    must_fail(lambda: check_seeded(_unrestored), "SEED without restoring the stream")
