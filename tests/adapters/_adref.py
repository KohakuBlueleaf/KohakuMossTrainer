"""Independent fp64 references for adapter math.

`lycoris_delta` restates LyCORIS semantics from a layer's saved tensors; `comfy_weight`
restates how ComfyUI's weight adapters patch a weight at strength 1 (alpha/rank rules,
output-axis DoRA normalised by the un-merged weight, OFT merge ignoring `rescale`).
Neither calls LyCORIS or KohakuMoss code.
"""

import torch
from torch import Tensor

EPS32 = torch.finfo(torch.float32).eps


def _d(t: Tensor) -> Tensor:
    return t.detach().double().cpu()


def _cayley(blocks: Tensor, cap: float) -> Tensor:
    q = blocks - blocks.transpose(-1, -2)
    if cap > 0:
        norm = q.norm() + 1e-8
        if norm > cap:
            q = q * (cap / norm)
    eye = torch.eye(q.shape[-1], dtype=q.dtype)
    return (eye + q) @ torch.linalg.inv(eye - q)


def _blockdiag_rotate(r: Tensor, w: Tensor) -> Tensor:
    """W' rows of block k = R_k^T applied on the row axis: W'[k, m] = sum_n R[k, n, m] W[k, n]."""
    k, s, _ = r.shape
    out = torch.empty_like(w)
    for b in range(k):
        rows = slice(b * s, (b + 1) * s)
        out[rows] = r[b].transpose(0, 1) @ w[rows]
    return out


def _butterfly(r: Tensor, w: Tensor) -> Tensor:
    """BOFT: stage i pairs rows i·b/2 apart inside groups of 2^i·b, as index permutations."""
    m, num, b, _ = r.shape
    rows = w.shape[0]
    half = b // 2
    x = w.clone()
    for i in range(m):
        k = (2**i) * half
        idx = torch.arange(rows).reshape(-1, 2, k).transpose(1, 2).reshape(-1)
        perm = x[idx]
        perm = perm.reshape(num, b, -1)
        perm = torch.einsum("bij,bj...->bi...", r[i], perm).reshape(rows, -1)
        out = torch.empty_like(x)
        out[idx] = perm
        x = out
    return x


def _kron_sides(t: dict) -> tuple[Tensor, Tensor, int | None]:
    rank = None
    if "lokr_w1" in t:
        w1 = _d(t["lokr_w1"])
    else:
        w1 = _d(t["lokr_w1_a"]) @ _d(t["lokr_w1_b"])
        rank = t["lokr_w1_b"].shape[0]
    if "lokr_w2" in t:
        w2 = _d(t["lokr_w2"])
    else:
        w2 = _d(t["lokr_w2_a"]) @ _d(t["lokr_w2_b"])
        rank = t["lokr_w2_b"].shape[0]
    return w1, w2, rank


def lycoris_delta(algo: str, t: dict, w: Tensor, out_dim: int) -> Tensor:
    """ΔW of a layer from its native LyCORIS tensors (multiplier 1)."""
    w = _d(w)
    match algo:
        case "locon":
            dw = (
                float(t["alpha"])
                / t["lora_down.weight"].shape[0]
                * (_d(t["lora_up.weight"]) @ _d(t["lora_down.weight"]))
            )
        case "loha":
            r = t["hada_w1_b"].shape[0]
            dw = (float(t["alpha"]) / r) * (
                (_d(t["hada_w1_a"]) @ _d(t["hada_w1_b"]))
                * (_d(t["hada_w2_a"]) @ _d(t["hada_w2_b"]))
            )
        case "lokr":
            w1, w2, rank = _kron_sides(t)
            scale = 1.0 if rank is None else float(t["alpha"]) / rank
            dw = torch.kron(w1, w2) * scale
        case "glora":
            r = t["a2.weight"].shape[0]
            dw = (float(t["alpha"]) / r) * (
                w @ _d(t["a1.weight"]) @ _d(t["a2.weight"])
                + _d(t["b1.weight"]) @ _d(t["b2.weight"])
            )
        case "oft":
            r = _cayley(_d(t["oft_blocks"]), float(t["alpha"]) * out_dim)
            wp = _blockdiag_rotate(r, w)
            if "rescale" in t:
                wp = _d(t["rescale"]).reshape(-1, 1) * wp
            dw = wp - w
        case "boft":
            r = _cayley(_d(t["oft_blocks"]), float(t["alpha"]) * out_dim)
            wp = _butterfly(r, w)
            if "rescale" in t:
                wp = _d(t["rescale"]).reshape(-1, 1) * wp
            dw = wp - w
        case "ia3":
            v = _d(t["weight"])
            dw = w * (v[None, :] if int(t["on_input"]) else v[:, None])
        case "full":
            dw = _d(t["diff"])
        case _:
            raise KeyError(algo)
    if "dora_scale" in t:
        merged = w + dw
        m = _d(t["dora_scale"])
        if m.shape[0] == merged.shape[0]:
            norm = merged.norm(dim=1, keepdim=True)
        else:
            norm = merged.norm(dim=0, keepdim=True)
        dw = merged * (m / (norm + EPS32)) - w
    return dw


def comfy_weight(t: dict, w: Tensor) -> Tensor:
    """ComfyUI's patched weight at strength 1 for one layer's tensors."""
    w = _d(w)
    alpha = float(t["alpha"]) if "alpha" in t else None
    if "diff" in t:
        return w + _d(t["diff"])
    if "w_norm" in t:
        return w + _d(t["w_norm"])
    if "oft_blocks" in t and t["oft_blocks"].dim() == 3:
        r = _cayley(_d(t["oft_blocks"]), alpha or 0.0)
        return _blockdiag_rotate(r, w)
    if "oft_blocks" in t:
        wp = _butterfly(_cayley(_d(t["oft_blocks"]), alpha or 0.0), w)
        if "rescale" in t:
            wp = wp * _d(t["rescale"]).reshape(-1, 1)
        return wp
    if "lora_up.weight" in t:
        rank = t["lora_down.weight"].shape[0]
        raw = _d(t["lora_up.weight"]) @ _d(t["lora_down.weight"])
    elif "hada_w1_a" in t:
        rank = t["hada_w1_b"].shape[0]
        raw = (_d(t["hada_w1_a"]) @ _d(t["hada_w1_b"])) * (
            _d(t["hada_w2_a"]) @ _d(t["hada_w2_b"])
        )
    elif "a1.weight" in t:
        rank = t["a2.weight"].shape[0]
        raw = w @ _d(t["a1.weight"]) @ _d(t["a2.weight"]) + _d(t["b1.weight"]) @ _d(
            t["b2.weight"]
        )
    else:
        w1, w2, rank = _kron_sides(t)
        raw = torch.kron(w1, w2)
    scale = 1.0 if alpha is None or rank is None else alpha / rank
    calc = w + scale * raw
    if "dora_scale" not in t:
        return calc
    m = _d(t["dora_scale"])
    if m.shape[0] == calc.shape[0]:
        norm = w.norm(dim=1, keepdim=True)
    else:
        norm = calc.norm(dim=0, keepdim=True)
    return calc * (m / (norm + EPS32))
