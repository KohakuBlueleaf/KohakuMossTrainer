"""The text-block FFN's SiTU pair equals the eager spec on the two linears' outputs
(model-free)."""

import pytest
import torch

from kohakumoss.model.components.ffn import situ_pair
from kohakumoss_kernels.backends.eager.ffn import situ_glu
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _check(cond):
    assert cond


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_situ_pair_equals_spec(dtype):
    g = torch.Generator().manual_seed(0)
    u = (torch.randn(13, 40, generator=g) * 30).to(dtype)
    gate = (torch.randn(13, 40, generator=g) * 6).to(dtype)
    tol = 1e-12 if dtype == torch.float64 else 5e-3
    ref = situ_glu(torch.cat((u, gate), -1))
    assert _rel(situ_pair(u, gate), ref) < tol
    must_fail(lambda: _check(_rel(situ_pair(gate, u), ref) < tol), "up / gate swapped")


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()
