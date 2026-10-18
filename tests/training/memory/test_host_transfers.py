"""Host-to-device handover without blocking copies, model-free: the loader pins batches
for any accelerator with pinned host memory (XPU as well as CUDA), and the token
denominator's in-place reduction never writes into the caller's count tensor."""

import pytest
import torch

from kohakumoss.diffusion.objective.normalize import TokenNormalizer
from kohakumoss_common.devices import streamed_type
from kohakumoss_data.loader import stateless
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _accelerator(monkeypatch, kind: str | None) -> None:
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: kind is not None)
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda: torch.device(kind or "cpu")
    )


def test_loader_pins_for_xpu(monkeypatch):
    _accelerator(monkeypatch, "xpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert stateless.pinnable() and streamed_type("xpu") and streamed_type("cuda")
    must_fail(lambda: _yes(torch.cuda.is_available()), "pinned only on CUDA")
    _accelerator(monkeypatch, None)
    assert not stateless.pinnable()


def _yes(flag: bool) -> None:
    assert flag


def _add_one(t: torch.Tensor) -> torch.Tensor:
    return t.add_(1)


def check_untouched(normalizer: TokenNormalizer) -> None:
    count = torch.tensor(5, dtype=torch.int64)
    denom = normalizer.denominator(count, "cpu")
    assert int(denom) == 6 and int(count) == 5, (int(denom), int(count))
    assert int(normalizer.denominator(7, "cpu")) == 8


def _shared(self, local_count, device) -> torch.Tensor:
    t = torch.as_tensor(local_count, dtype=torch.int64).reshape(1)
    return self.reduce(t).clamp_min(1)


def test_denominator_reduces_a_copy(monkeypatch):
    check_untouched(TokenNormalizer(_add_one))
    monkeypatch.setattr(TokenNormalizer, "denominator", _shared)
    must_fail(lambda: check_untouched(TokenNormalizer(_add_one)), "caller's count")
