"""`_master.stacked`: [a; b] in the activation dtype, the operand of the single
`linear_qk_rope` input-gradient GEMM, equal to torch.cat then cast."""

import pytest
import torch

from kohakumoss_kernels.backends._master import stacked
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_stacked(stack) -> None:
    wq, wk = torch.randn(6, 5), torch.randn(6, 5)
    like = torch.empty(3, 12, dtype=torch.bfloat16)
    got = stack(wq, wk, like)
    want = torch.cat((wq, wk)).to(torch.bfloat16)
    assert got.dtype == torch.bfloat16 and got.is_contiguous()
    assert torch.equal(got, want)


def test_stacked_matches_cat():
    check_stacked(stacked)


def test_negative_swapped_halves():
    must_fail(lambda: check_stacked(lambda a, b, like: stacked(b, a, like)), "swapped")
