"""DDP_REDUCE_DTYPE 'bf16': the knob maps to bfloat16; an fp32 bucket reduces on a bf16
copy that is marked narrow (not capped by WIDE_AHEAD), a 16-bit bucket under fp32 is
wide (capped), and the copy round-trips into the fp32 gradients."""

import pytest
import torch
from torch import nn

from kohakumoss.parallel.ddp import _Bucket
from kohakumoss.training.build.parallel import reduce_dtype
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


def _check(narrow_of) -> None:
    fp32 = [nn.Parameter(torch.ones(8)), nn.Parameter(torch.ones(4))]
    half = [nn.Parameter(torch.ones(8, dtype=torch.float16))]
    assert narrow_of(_Bucket(fp32, torch.bfloat16)) is True
    assert narrow_of(_Bucket(half, torch.float32)) is False
    assert narrow_of(_Bucket(fp32, torch.float32)) is False


def test_bucket_narrow():
    _check(lambda b: b.narrow)


def test_bucket_narrow_negative():
    must_fail(lambda: _check(lambda b: b.reduce_dtype is not None), "any copy capped")


def test_bf16_round_trip():
    p = nn.Parameter(torch.zeros(6))
    b = _Bucket([p], torch.bfloat16)
    b.attach()
    p.grad.copy_(torch.tensor([1.0, -2.0, 0.5, 3.0, 0.25, 4.0]))
    buf = b.reduced_buffer()
    assert buf.dtype == torch.bfloat16
    buf.mul_(2)
    b.copy_back()
    assert torch.equal(p.grad, torch.tensor([2.0, -4.0, 1.0, 6.0, 0.5, 8.0]))
    assert b.reduced_bytes == 6 * 2


def test_knob():
    assert reduce_dtype("bf16") is torch.bfloat16
    assert reduce_dtype("fp32") is torch.float32 and reduce_dtype("param") is None
    with pytest.raises(ValueError):
        reduce_dtype("fp16")
