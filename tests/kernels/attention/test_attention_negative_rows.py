"""Attention rows whose largest logit is far below zero, on segments that end inside a
key tile: dQ / dK / dV stay finite (padded keys carry no probability) and no worse than
PyTorch SDPA against fp64."""

import pytest
import torch

from tests._util.negative import must_fail
from tests.kernels.attention._attention_case import (
    KERNELS,
    check_stats,
    kernel_fn,
    pytorch_sdpa,
    ref64,
    run,
)

pytestmark = pytest.mark.gpu

DTYPES = (torch.float16, torch.bfloat16)
LENS = [700, 333, 77]


def negative_case(device, dtype, seed=0):
    """q clustered at +c, k at -c (every scaled logit ~ -100: exp2 of a padded key
    overflows fp32), dO ~ N(0, 1)."""
    g = torch.Generator(device=device).manual_seed(seed)
    n, heads, d = sum(LENS), 3, 128
    cu = torch.tensor(
        [0, *torch.tensor(LENS).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    q, k, v, do = (
        torch.randn(n, heads, d, device=device, generator=g) for _ in range(4)
    )
    center = torch.randn(1, heads, d, device=device, generator=g)
    q = (q * 0.05 + center) * 3.0
    k = (k * 0.05 - center) * 3.0
    return (*(t.to(dtype) for t in (q, k, v, do)), cu, max(LENS))


def check_finite(grads) -> None:
    for name, g in zip(("dq", "dk", "dv"), grads):
        assert torch.isfinite(g).all(), f"non-finite {name}"


@pytest.mark.parametrize("backend", KERNELS)
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
def test_negative_max_logit_rows(backend, dtype, device):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(False):
        pytest.skip("no bf16 on this card")
    fn = kernel_fn(backend, device, dtype)
    case = negative_case(device, dtype)
    out, grads = run(fn, *case)
    check_finite(grads)
    eo, eg = pytorch_sdpa(*case)
    ro, rg = ref64(*case)
    check_stats(backend, out, grads, eo, eg, ro, rg, dtype)
    nan = [g.clone() for g in grads]
    nan[0].view(-1)[3] = float("nan")
    must_fail(lambda: check_finite(nan), "one NaN in dQ")
