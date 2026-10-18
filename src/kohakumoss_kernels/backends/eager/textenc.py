"""Text-encoder ops (Qwen3.5 Gated DeltaNet, causal GQA attention): the eager reference.

Every op reduces in at least fp32 (fp64 for fp64 inputs) and rounds once to the input
activation dtype. Packed `[N, ...]` tensors with segment boundaries `cu_seqlens`;
nothing crosses a boundary. See docs/internals/text-encoder.md.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_kernels.backends.eager._acc import acc_dtype

__all__ = [
    "causal_conv_varlen",
    "gated_delta_rule_varlen",
    "rms_norm_gated",
    "attention_varlen_causal",
]

CHUNK = 64
SUB = 16
L2_EPS = 1e-6


def segment_positions(cu_seqlens: Tensor, n: int) -> Tensor:
    """Position of every packed token inside its segment, int64 [N]."""
    idx = torch.arange(n, device=cu_seqlens.device)
    cu = cu_seqlens.to(torch.int64)
    seg = torch.searchsorted(cu[1:], idx, right=True)
    return idx - cu.index_select(0, seg)


def causal_conv_varlen(x: Tensor, weight: Tensor, cu_seqlens: Tensor) -> Tensor:
    """silu(sum_j weight[:, W-1-j] * x[t-j]) over taps inside the token's segment."""
    dt = acc_dtype(x, weight)
    xf, w = x.to(dt), weight.to(dt)
    n, width = x.shape[0], weight.shape[1]
    pos = segment_positions(cu_seqlens, n)
    out = xf * w[:, width - 1]
    for j in range(1, width):
        shifted = F.pad(xf, (0, 0, j, 0))[:n]
        keep = (pos >= j).unsqueeze(1)
        out = out + torch.where(keep, shifted, 0.0) * w[:, width - 1 - j]
    return F.silu(out).to(x.dtype)


def rms_norm_gated(x: Tensor, z: Tensor, weight: Tensor, eps: float) -> Tensor:
    """`x / rms(x) * weight * silu(z)` over the last dim."""
    dt = acc_dtype(x, z)
    xf = x.to(dt)
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (y * weight.to(dt) * F.silu(z.to(dt))).to(x.dtype)


def attention_varlen_causal(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    scale: float,
) -> Tensor:
    """Causal attention inside each segment; query head h reads kv head h // groups."""
    bounds = cu_seqlens.tolist()
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        qs, ks, vs = (t[start:end].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        ks = ks.repeat_interleave(groups, dim=1)
        vs = vs.repeat_interleave(groups, dim=1)
        o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale)
        out[start:end] = o.squeeze(0).transpose(0, 1)
    return out


def _to_padded(x: Tensor, seg: Tensor, pos: Tensor, b: int, length: int) -> Tensor:
    """[N, *rest] -> [B, length, *rest] with zeros past each segment's end."""
    out = x.new_zeros((b, length) + tuple(x.shape[1:]))
    out[seg, pos] = x
    return out


def _mm(x: Tensor, y: Tensor, it: torch.dtype) -> Tensor:
    """x @ y on operands in the input dtype `it`; PyTorch's matmul, its output in `it`."""
    return x.to(it) @ y.to(it)


def _pair_decay(g: Tensor) -> Tensor:
    """D[..., i, j] = sum_{j < t <= i} g_t over the last axis (0 for i <= j)."""
    n = g.shape[-1]
    later = torch.ones(n, n, dtype=torch.bool, device=g.device).tril(-1)
    return (g.unsqueeze(-1) * later).cumsum(-2)


def _ut_inverse(a: Tensor, it: torch.dtype) -> Tensor:
    """(I + a)^-1 of strictly lower [..., 64, 64] blocks: 16 x 16 diagonal inverses in
    fp32 vector math, off-diagonal blocks by block substitution with `it` products."""
    s, nb = SUB, a.shape[-1] // SUB
    eye = torch.eye(s, dtype=a.dtype, device=a.device)
    blk = [[a[..., i * s : (i + 1) * s, j * s : (j + 1) * s] for j in range(nb)] for i in range(nb)]  # fmt: skip
    t = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        t[i][i] = torch.linalg.solve_triangular(
            blk[i][i] + eye, eye.expand_as(blk[i][i]), upper=False, unitriangular=True
        )
    for i in range(1, nb):
        for j in range(i - 1, -1, -1):
            acc = sum(_mm(blk[i][m], t[m][j], it) for m in range(j, i))
            t[i][j] = -_mm(t[i][i], acc, it)
    zero = torch.zeros_like(blk[0][0])
    rows = [torch.cat([t[i][j] if j <= i else zero for j in range(nb)], -1) for i in range(nb)]  # fmt: skip
    return torch.cat(rows, -2)


def _chunked(q, k, v, g, beta, scale: float, it: torch.dtype) -> Tensor:
    """Chunked gated delta rule on padded [B, H, L, D] raw q / k (L a multiple of 64);
    every matmul takes `it` operands (raw rows exact)."""
    b, h, length, dk = k.shape
    dv = v.shape[-1]
    nc = length // CHUNK
    q, k, v = (t.reshape(b, h, nc, CHUNK, t.shape[-1]) for t in (q, k, v))
    g = g.reshape(b, h, nc, CHUNK)
    beta = beta.reshape(b, h, nc, CHUNK)
    rk = torch.rsqrt(k.square().sum(-1) + L2_EPS)
    rq = torch.rsqrt(q.square().sum(-1) + L2_EPS) * scale
    d = _pair_decay(g)
    low = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).tril()
    strict = low.tril(-1)
    kk = _mm(k, k.transpose(-1, -2), it) * rk.unsqueeze(-1) * rk.unsqueeze(-2)
    a = torch.where(strict, kk * beta.unsqueeze(-1) * d.exp(), 0.0)
    t = _ut_inverse(a, it)
    qk = _mm(q, k.transpose(-1, -2), it) * rq.unsqueeze(-1) * rk.unsqueeze(-2)
    qk = torch.where(low, qk * d.exp(), 0.0)
    gc = g.cumsum(-1)
    suffix = (g.unsqueeze(-1) * strict).sum(-2)
    fy = (rk * beta * gc.exp()).unsqueeze(-1)
    fq = (rq * gc.exp()).unsqueeze(-1)
    fk = (rk * suffix.exp()).unsqueeze(-1)
    tail = g.sum(-1).exp()
    state = q.new_zeros(b, h, dk, dv)
    out = torch.empty_like(v)
    for c in range(nc):
        kc, qc = k[:, :, c], q[:, :, c]
        y = v[:, :, c] * beta[:, :, c].unsqueeze(-1) - fy[:, :, c] * _mm(kc, state, it)
        vn = _mm(t[:, :, c], y, it)
        out[:, :, c] = fq[:, :, c] * _mm(qc, state, it) + _mm(qk[:, :, c], vn, it)
        kd = fk[:, :, c] * vn
        state = state * tail[:, :, c, None, None] + _mm(kc.transpose(-1, -2), kd, it)
    return out.reshape(b, h, length, dv)


def gated_delta_rule_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """Gated delta rule per segment (see kohakumoss_kernels.ops), chunked: vector math
    in fp32+ (fp64 for fp64 inputs), every matmul a plain PyTorch matmul on operands in
    the input dtype."""
    dt = acc_dtype(q, k, v)
    it = q.dtype
    n, hk, _ = q.shape
    hv = v.shape[1]
    groups = hv // hk
    g = -a_log.to(dt).exp() * F.softplus(a.to(dt) + dt_bias.to(dt))
    beta = torch.sigmoid(b.to(dt))
    qn = q.to(dt).repeat_interleave(groups, dim=1)
    kn = k.to(dt).repeat_interleave(groups, dim=1)
    nseg = cu_seqlens.shape[0] - 1
    length = max(-(-int(max_seqlen) // CHUNK), 1) * CHUNK
    pos = segment_positions(cu_seqlens, n)
    seg = torch.searchsorted(
        cu_seqlens[1:].to(torch.int64), torch.arange(n, device=q.device), right=True
    )

    def pad(t: Tensor) -> Tensor:
        return _to_padded(t, seg, pos, nseg, length).transpose(1, 2)

    out = _chunked(
        pad(qn), pad(kn), pad(v.to(dt)), pad(g), pad(beta), q.shape[-1] ** -0.5, it
    )
    return out.transpose(1, 2)[seg, pos].to(v.dtype)
