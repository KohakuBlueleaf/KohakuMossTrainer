"""Shared helpers of the attention tests: kernel lookup, cases, references, the error
statistics check."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager.attention import attention_varlen as eager
from kohakumoss_kernels.error_stats import KEYS, compare
from tests._util.error_stats import record, regressions

KERNELS = ("triton", "tilelang")


def kernel_fn(backend: str, device: torch.device, dtype: torch.dtype):
    """The backend's own kernels (not the per-shape choice), or skip when they cannot
    serve here."""
    reason = registry.available("attention_varlen", device, dtype).get(backend)
    if reason is not None:
        pytest.skip(f"{backend}: {reason}")
    if backend == "tilelang" and device.type != "cuda":
        pytest.skip("tilelang is CUDA only")
    module = importlib.import_module(
        f"kohakumoss_kernels.backends.{backend}.attention.varlen"
    )
    return module.attention_varlen


def make_case(lens, device, heads=3, d=128, dtype=torch.float16, qk_scale=1.0, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    n = sum(lens)
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    q, k, v, do = (
        torch.randn(n, heads, d, device=device, generator=g).to(dtype) for _ in range(4)
    )
    if qk_scale != 1.0:
        center = torch.randn(1, heads, d, device=device, generator=g)
        q = ((q.float() * 0.05 + center) * qk_scale).to(dtype)
        k = ((k.float() * 0.05 + center) * qk_scale).to(dtype)
    return q, k, v, do, cu, max(lens)


def run(fn, q, k, v, do, cu, max_len):
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    out = fn(*leaves, cu, max_len, q.shape[-1] ** -0.5)
    out.backward(do)
    return out.detach(), [t.grad for t in leaves]


def ref64(q, k, v, do, cu, max_len):
    leaves = [t.double().requires_grad_() for t in (q, k, v)]
    out = eager(*leaves, cu, max_len, q.shape[-1] ** -0.5)
    out.backward(do.double())
    return out.detach(), [t.grad for t in leaves]


def pytorch_sdpa(q, k, v, do, cu, max_len):
    """Plain PyTorch: one `scaled_dot_product_attention` per segment in the inputs'
    16-bit dtype, gradients by autograd; (o, [dq, dk, dv])."""
    scale = q.shape[-1] ** -0.5
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    bounds = cu.tolist()
    pieces = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e == s:
            continue
        qs, ks, vs = (t[s:e].transpose(0, 1).unsqueeze(0) for t in leaves)
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        pieces.append(o[0].transpose(0, 1))
    if not pieces:
        return torch.zeros_like(q), [torch.zeros_like(t) for t in (q, k, v)]
    out = torch.cat(pieces)
    out.backward(do)
    return out.detach(), [t.grad for t in leaves]


def check_stats(name, out, grads, eo, eg, ro, rg, dtype=None, keys=KEYS):
    """The error statistics of o, dq, dk, dv against PyTorch's `eo`, `eg` and the fp64
    `ro`, `rg`: writes the stats table, fails only on a regression of a stat in `keys`.
    """
    for label, a, e, r in [("o", out, eo, ro), *zip(("dq", "dk", "dv"), grads, eg, rg)]:
        table = compare(a, e, r)
        record("attention", f"{name} {label}", table)
        bad = [k for k in regressions(table) if k in keys or k == "non-finite"]
        assert not bad, f"{name} {label}: regresses on {bad}: {table}"


def dropped(t, cu, seg: int = 0):
    """Broken variant: segment `seg` of the packed tensor left as zeros."""
    out = t.clone()
    s, e = cu[seg].item(), cu[seg + 1].item()
    out[s:e] = 0
    return out


def cosine(a, b) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def unshifted(q, k, v, cu, max_len, scale):
    """Broken variant: softmax without the row-max shift (exp overflows at large logits)."""
    bounds = cu.tolist()
    out = torch.empty_like(q)
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e == s:
            continue
        logits = torch.einsum("qhd,khd->hqk", q[s:e].float(), k[s:e].float()) * scale
        p = logits.exp()
        p = p / p.sum(-1, keepdim=True)
        out[s:e] = torch.einsum("hqk,khd->qhd", p, v[s:e].float()).to(q.dtype)
    return out


def unguarded(q, k, v, cu, max_len, scale):
    """Broken variant: padded attention whose empty segments' rows (every key masked)
    are not guarded; their NaN gradients reach the shared padding row of K / V."""
    b = cu.numel() - 1
    lens = cu[1:] - cu[:-1]
    pos = torch.arange(max(max_len, 1), device=q.device)
    idx = torch.where(pos[None] < lens[:, None], cu[:-1, None] + pos[None], 0).long()
    keep = pos[None] < lens[:, None]
    qp, kp, vp = (t[idx].transpose(1, 2) for t in (q, k, v))
    mask = keep[:, None, None, :].expand(b, 1, idx.shape[1], idx.shape[1])
    logits = (qp.float() @ kp.float().transpose(-1, -2)) * scale
    p = logits.masked_fill(~mask, float("-inf")).softmax(-1)
    o = (p @ vp.float()).to(q.dtype).transpose(1, 2)
    out = torch.zeros_like(q).index_put((idx[keep],), o[keep])
    return out
