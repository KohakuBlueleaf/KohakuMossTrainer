"""The text prewarm twin's tensors take the dtype they compute in: a 16-bit float held in
another 16-bit dtype (a host embedding kept bf16 on an fp16 card) becomes the compute
dtype; fp32, integer and same-dtype tensors keep theirs (pure function)."""

import pytest
import torch

from kohakumoss.encoders.text.prewarm import _computed
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


def check_computed(computed) -> None:
    held = torch.zeros(4, 3, dtype=torch.bfloat16)
    out = computed(held, torch.float16)
    assert out.dtype == torch.float16 and out.shape == held.shape
    for t in (
        torch.zeros(3, dtype=torch.float16),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.int8),
    ):
        assert computed(t, torch.float16) is t


def test_held_16bit_tensors_take_the_compute_dtype():
    check_computed(_computed)
    must_fail(lambda: check_computed(lambda t, d: t), "the stored dtype kept")
