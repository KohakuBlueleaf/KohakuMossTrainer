"""The eager `attention_varlen` reference against an independent dense block-diagonal
softmax in fp64: output and q / k / v gradients, empty segments, rows past the last
bound."""

import pytest
import torch

from kohakumoss_kernels.backends.eager.attention import attention_varlen
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def dense(q, k, v, cu, n, scale):
    """softmax(scale q k^T + block mask) v over the packed rows; rows outside every
    segment are zero."""
    seg = torch.full((n,), -1, dtype=torch.long)
    bounds = cu.tolist()
    for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        seg[s:e] = i
    same = (seg[:, None] == seg[None, :]) & (seg[:, None] >= 0)
    logits = torch.einsum("qhd,khd->hqk", q, k) * scale
    p = logits.masked_fill(~same, float("-inf")).softmax(-1).nan_to_num(0.0)
    return torch.einsum("hqk,khd->qhd", p, v)


def case(lens, tail=0, heads=2, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(lens) + tail
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    q, k, v, do = (
        torch.randn(n, heads, d, generator=g, dtype=torch.float64) for _ in range(4)
    )
    return q, k, v, do, cu, n


def outputs(fn, q, k, v, do, cu, n):
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = fn(*leaves, cu, n, q.shape[-1] ** -0.5)
    out.backward(do)
    return [out.detach(), *(t.grad for t in leaves)]


def check(got, want):
    for a, b, name in zip(got, want, ("o", "dq", "dk", "dv")):
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-12), name


@pytest.mark.parametrize(
    "lens,tail", [([5, 0, 9, 1, 3], 0), ([0, 7, 0], 0), ([4, 6], 3), ([11], 0)]
)
def test_eager_matches_dense(lens, tail):
    q, k, v, do, cu, n = case(lens, tail)
    want = outputs(lambda *a: dense(*a[:4], n, a[5]), q, k, v, do, cu, n)
    check(outputs(attention_varlen, q, k, v, do, cu, n), want)


def test_eager_check_fails_across_segments():
    q, k, v, do, cu, n = case([5, 9, 3])
    want = outputs(lambda *a: dense(*a[:4], n, a[5]), q, k, v, do, cu, n)
    leak = torch.tensor([0, n], dtype=torch.int32)
    got = outputs(attention_varlen, q, k, v, do, leak, n)
    must_fail(lambda: check(got, want), "one segment over every row")
