"""ComfyUI's fused joint-attention layout (`layers.N.proj.qkv`, q;k;v rows): per block,
the q / k / v adapters as one layer, and its split back into q / k / v. See
docs/guides/export-to-comfy.md#fused-qkv."""

import re
from collections.abc import Iterable

import torch
from torch import Tensor

from kohakumoss.adapters.infer.formats import split_key

PARTS = ("q", "k", "v")
JOINT = re.compile(r"^(layers\.\d+\.proj)\.([qkv])$")
FUSED_STEM = re.compile(r"(?:^|[._])layers[._]\d+[._]proj[._]qkv$")
LORA = frozenset({"lora_up.weight", "lora_down.weight", "alpha"})


def qkv_groups(paths: Iterable[str]) -> dict[str, dict[str, str]]:
    """{`layers.N.proj.qkv`: {"q" | "k" | "v": layer path}} of the joint-attention
    layers among `paths`."""
    out: dict[str, dict[str, str]] = {}
    for path in paths:
        found = JOINT.match(path)
        if found:
            out.setdefault(f"{found.group(1)}.qkv", {})[found.group(2)] = path
    return out


def is_lora(tensors: dict) -> bool:
    """A plain LoRA set (up, down, optional alpha), nothing else."""
    return {"lora_up.weight", "lora_down.weight"} <= set(tensors) <= LORA


def _scaled_up(parts: dict) -> tuple[Tensor, Tensor]:
    """(up * alpha / rank, down) in fp32; a missing alpha is scale 1."""
    up = parts["lora_up.weight"].float()
    down = parts["lora_down.weight"].float()
    alpha = parts.get("alpha")
    scale = 1.0 if alpha is None else float(alpha) / down.shape[0]
    return up * scale, down


def fused_lora(parts: dict[str, dict], rows: dict[str, int]) -> dict[str, Tensor]:
    """One LoRA of rank r_q + r_k + r_v: down = cat(downs), up block-diagonal with each
    layer's scale folded in, alpha = rank (scale 1); an absent layer has zero rows."""
    ups, downs = {}, []
    for name in PARTS:
        if name in parts:
            ups[name], down = _scaled_up(parts[name])
            downs.append(down)
    rank = sum(d.shape[0] for d in downs)
    up = downs[0].new_zeros(sum(rows.values()), rank)
    row = col = 0
    for name in PARTS:
        if name in ups:
            u = ups[name]
            up[row : row + u.shape[0], col : col + u.shape[1]] = u
            col += u.shape[1]
        row += rows[name]
    down = torch.cat(downs, 0)
    alpha = torch.tensor(float(rank))
    return {"lora_up.weight": up, "lora_down.weight": down, "alpha": alpha}


def fused_diff(
    parts: dict[str, dict], rows: dict[str, int], cols: int
) -> dict[str, Tensor]:
    """{"diff" (, "diff_b")}: each layer's `diff` rows in q, k, v order, zero rows for
    an absent layer."""
    like = next(iter(parts.values()))["diff"]
    blocks, biases = [], []
    any_bias = any("diff_b" in p for p in parts.values())
    for name in PARTS:
        p = parts.get(name)
        blocks.append(p["diff"].float() if p else like.new_zeros(rows[name], cols))
        if any_bias:
            b = None if p is None else p.get("diff_b")
            biases.append(like.new_zeros(rows[name]) if b is None else b.float())
    out = {"diff": torch.cat(blocks, 0)}
    if any_bias:
        out["diff_b"] = torch.cat(biases, 0)
    return out


def _split_lora(t: dict) -> list[dict | None]:
    """Each third of the fused rows as its own LoRA over the down rows its up uses
    (alpha kept per rank, so each part's scale is the fused one); None for a third
    with no nonzero up column."""
    up, down = t["lora_up.weight"], t["lora_down.weight"]
    alpha = t.get("alpha")
    rank = down.shape[0]
    out = []
    for part in up.chunk(3, 0):
        used = part.abs().sum(0).nonzero().flatten()
        if not used.numel():
            out.append(None)
            continue
        p = {"lora_up.weight": part[:, used], "lora_down.weight": down[used]}
        if alpha is not None:
            p["alpha"] = torch.tensor(float(alpha) * used.numel() / rank)
        out.append(p)
    return out


def _split_diff(t: dict) -> list[dict | None]:
    out = [{"diff": d} for d in t["diff"].chunk(3, 0)]
    if "diff_b" in t:
        for p, b in zip(out, t["diff_b"].chunk(3, 0)):
            p["diff_b"] = b
    return out


def split_qkv(tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    """`tensors` with every fused joint-attention layer (a LoRA or a `diff`) split into
    q / k / v layers under the same key style; other keys unchanged. Raises KeyError
    for a fused layer of another kind."""
    fused: dict[str, dict[str, Tensor]] = {}
    out = {}
    for key, value in tensors.items():
        parts = split_key(key)
        if parts is not None and FUSED_STEM.search(parts[0]):
            fused.setdefault(parts[0], {})[parts[1]] = value
        else:
            out[key] = value
    for stem, t in fused.items():
        match t:
            case _ if is_lora(t):
                split = _split_lora(t)
            case {"diff": _}:
                split = _split_diff(t)
            case _:
                raise KeyError(
                    f"{stem}: a fused q/k/v layer with {sorted(t)} cannot be split; "
                    "only LoRA and diff fused layers load"
                )
        for name, p in zip(PARTS, split):
            if p is not None:
                out.update({f"{stem[:-3]}{name}.{k}": v for k, v in p.items()})
    return out
