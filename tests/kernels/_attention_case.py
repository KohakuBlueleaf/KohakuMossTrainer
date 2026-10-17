"""Shared helpers of the attention tests: kernel lookup, cases, references, bars."""

import importlib

import pytest
import torch

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.eager.attention import attention_varlen as eager
from tests._util.precision import assert_bar

KERNELS = ("triton", "tilelang")


def kernel_fn(backend: str, device: torch.device, dtype: torch.dtype):
    """The backend's own kernels (not the race), or skip when they cannot serve here."""
    reason = registry.available("attention_varlen", device, dtype).get(backend)
    if reason is not None:
        pytest.skip(f"{backend}: {reason}")
    if backend == "tilelang" and device.type != "cuda":
        pytest.skip("tilelang is CUDA only")
    module = importlib.import_module(f"kohakumoss_kernels.backends.{backend}.attention")
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


def mm16(q, k, v, do, cu, max_len):
    """PyTorch at the kernels' matmul precision: per segment, 16-bit operands with fp32
    accumulation (fp32 matmuls of 16-bit values), fp32 softmax, P and dS rounded to the
    input dtype before their matmuls, delta from the rounded O. Returns (o, [dq, dk, dv])."""
    dt, scale = q.dtype, q.shape[-1] ** -0.5
    out, grads = torch.zeros_like(q), [torch.zeros_like(t) for t in (q, k, v)]
    bounds = cu.tolist()
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e == s:
            continue
        qf, kf, vf, dof = (t[s:e].transpose(0, 1).float() for t in (q, k, v, do))
        p = (qf @ kf.transpose(-1, -2) * scale).softmax(-1)
        p16 = p.to(dt).float()
        o = (p16 @ vf).to(dt)
        dp = dof @ vf.transpose(-1, -2)
        delta = (dof * o.float()).sum(-1, keepdim=True)
        ds = (p * (dp - delta)).to(dt).float()
        parts = (ds @ kf * scale, ds.transpose(-1, -2) @ qf * scale, p16.mT @ dof)
        out[s:e] = o.transpose(0, 1)
        for g, part in zip(grads, parts):
            g[s:e] = part.to(dt).transpose(0, 1)
    return out, grads


def check_bar(name, out, grads, eo, eg, ro, rg, dtype=None):
    """The precision bar (`assert_bar`: finite, and not both abs and rel error over
    `FACTOR` x PyTorch's) on o, dq, dk, dv against PyTorch's `eo`, `eg`."""
    for label, a, e, r in [("o", out, eo, ro), *zip(("dq", "dk", "dv"), grads, eg, rg)]:
        assert_bar(a, e, r, f"{name} {label}")


def less_precise(e, r, dtype, factor=3.0):
    """Broken variant: a result whose error is `factor` x the eager error everywhere."""
    return (r + factor * (e.double() - r)).to(dtype)


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
