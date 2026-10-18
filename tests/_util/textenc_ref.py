"""Independent fp64 references for the text-encoder ops (token recurrence, dense masks),
and the whole-encoder check against transformers (no regression)."""

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_kernels.error_stats import stats
from tests._util.error_stats import assert_no_regression


def encode_error(out: list[Tensor], ref: list[Tensor]) -> dict[str, float]:
    """Error stats (kohakumoss_kernels.error_stats) of a per-prompt encode against its
    per-prompt fp64 reference, over every token."""
    assert len(out) == len(ref)
    return stats(torch.cat(out), torch.cat(ref))


def assert_encode_no_regression(
    out: list[Tensor], base: list[Tensor], ref: list[Tensor], name: str = ""
) -> dict:
    """The error-statistics check with transformers' encode `base` (same precision) as
    the baseline: records the stats table and fails only when `out` regresses vs the
    fp64 `ref`."""
    return assert_no_regression(
        torch.cat(out), torch.cat(base), torch.cat(ref), name, test="textenc"
    )


def delta_rule_recurrent(q, k, v, a, b, a_log, dt_bias, cu_seqlens) -> Tensor:
    """Gated delta rule, one token at a time per segment, in fp64."""
    q, k, v, a, b = (t.double() for t in (q, k, v, a, b))
    a_log, dt_bias = a_log.double(), dt_bias.double()
    hk, hv = q.shape[1], v.shape[1]
    groups = hv // hk
    g = -a_log.exp() * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    qn = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    qn = qn.repeat_interleave(groups, dim=1)
    kn = kn.repeat_interleave(groups, dim=1)
    out = torch.empty_like(v)
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        state = v.new_zeros(hv, k.shape[-1], v.shape[-1])
        for t in range(start, end):
            state = state * g[t].exp()[:, None, None]
            kv = torch.einsum("hkv,hk->hv", state, kn[t])
            delta = (v[t] - kv) * beta[t][:, None]
            state = state + kn[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hkv,hk->hv", state, qn[t])
    return out


def attention_causal_dense(q, k, v, cu_seqlens, scale) -> Tensor:
    """Causal block-diagonal attention through one dense fp64 mask."""
    q, k, v = q.double(), k.double(), v.double()
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    n = q.shape[0]
    idx = torch.arange(n, device=q.device)
    seg = torch.searchsorted(cu_seqlens[1:].long(), idx, right=True)
    mask = (seg[:, None] == seg[None, :]) & (idx[:, None] >= idx[None, :])
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    s = s.masked_fill(~mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), v)


def conv_dense(x, weight, cu_seqlens) -> Tensor:
    """Per-segment F.conv1d with zero left padding, then silu, in fp64."""
    x, w = x.double(), weight.double()
    out = torch.empty_like(x)
    bounds = cu_seqlens.tolist()
    width = w.shape[1]
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        seg = x[start:end].t().unsqueeze(0)
        y = F.conv1d(seg, w.unsqueeze(1), padding=width - 1, groups=w.shape[0])
        out[start:end] = F.silu(y[0, :, : end - start].t())
    return out
