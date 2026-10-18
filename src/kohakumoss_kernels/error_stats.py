"""Error statistics of a result against an fp64 reference, next to the same statistics
of the eager reference. See docs/internals/fp16.md#error-statistics.

Per tensor: cosine to the reference, |out - ref| and |out - ref| / |ref| (nonzero ref)
as mean / median / min / max.
"""

import torch
from torch import Tensor

KEYS = ("cos_deficit", "abs_mean", "abs_median", "rel_mean", "rel_median")
PAIRS = (("abs_mean", "rel_mean"), ("abs_median", "rel_median"))


def stats(out: Tensor, ref: Tensor) -> dict[str, float]:
    """Cosine to `ref` and the abs / rel error summaries of `out`."""
    o = out.detach().double().flatten()
    r = ref.detach().double().flatten()
    d = (o - r).abs()
    nz = r != 0
    rel = d[nz] / r[nz].abs() if bool(nz.any()) else d.new_zeros(1)
    finite = bool(torch.isfinite(o).all())
    cos = (
        torch.nn.functional.cosine_similarity(o, r, dim=0).item() if o.numel() else 1.0
    )
    row = {"finite": finite, "cos": cos, "cos_deficit": 1.0 - cos}
    for name, v in (("abs", d), ("rel", rel)):
        v = v if v.numel() else d.new_zeros(1)
        row[f"{name}_mean"] = v.mean().item()
        row[f"{name}_median"] = v.median().item()
        row[f"{name}_min"] = v.min().item()
        row[f"{name}_max"] = v.max().item()
    return row


def floors(out: Tensor, ref64: Tensor) -> dict[str, float]:
    """The rounding scale of `out`'s dtype per stat: eps x median |ref| for the abs
    stats, eps for the rel and cosine stats."""
    eps = torch.finfo(out.dtype).eps if out.is_floating_point() else 0.0
    scale = (
        ref64.detach().double().abs().flatten().median().item() if ref64.numel() else 0
    )
    return {
        "cos_deficit": eps,
        "abs_mean": eps * scale,
        "abs_median": eps * scale,
        "rel_mean": eps,
        "rel_median": eps,
    }


def compare(out: Tensor, eager: Tensor, ref64: Tensor) -> dict[str, dict]:
    """The stats table: {"kernel": stats(out), "eager": stats(eager), "floor": the
    rounding scale per stat} against `ref64`."""
    return {
        "kernel": stats(out, ref64),
        "eager": stats(eager, ref64),
        "floor": floors(out, ref64),
    }


def worse(table: dict[str, dict]) -> list[str]:
    """Every summary on which the kernel is worse than eager; "non-finite" first when
    the kernel's output is not finite and eager's is."""
    k, e = table["kernel"], table["eager"]
    out = [s for s in k if s not in ("finite", "cos") and s in e and k[s] > e[s]]
    broke = e.get("finite", True) and not k.get("finite", True)
    return ["non-finite", *out] if broke else out
