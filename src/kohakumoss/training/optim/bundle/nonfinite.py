"""Non-finite gradient report of a non-finite step: the first non-finite gradients by
parameter name, kind (adapter / base) and NaN / Inf counts. Built only on a non-finite
step (one host sync per scanned gradient there); finite steps pay nothing."""

import torch
from torch import Tensor

from kohakumoss.adapters.build.target import CHILD

LIMIT = 8


def kind_of(name: str) -> str:
    """`adapter` for a LyCORIS module parameter, else `base`."""
    return "adapter" if f".{CHILD}." in f".{name}" else "base"


def param_names(groups: list[dict]) -> list[str]:
    """One name per parameter of `groups`, in parameter order (`names` when the
    groups carry them, else `group<i>.param<j>`)."""
    out = []
    for gi, g in enumerate(groups):
        names = g.get("names") or [
            f"group{gi}.param{j}" for j in range(len(g["params"]))
        ]
        out += list(names)
    return out


def grad_report(
    named: list[tuple[str, Tensor | None]], limit: int = LIMIT
) -> list[dict]:
    """The first `limit` non-finite gradients of `named` (name, gradient or None)."""
    out = []
    for name, g in named:
        if g is None:
            continue
        nan, inf = (int(c) for c in torch.stack([g.isnan().sum(), g.isinf().sum()]))
        if nan or inf:
            out.append(
                {
                    "param": name,
                    "kind": kind_of(name),
                    "nan": nan,
                    "inf": inf,
                    "numel": g.numel(),
                }
            )
            if len(out) == limit:
                break
    return out


def nonfinite_count(grads: list[Tensor]) -> int:
    """How many of `grads` hold a NaN / Inf (one host sync)."""
    if not grads:
        return 0
    dev = grads[0].device
    bad = [(~torch.isfinite(g).all()).to(dev, torch.int32) for g in grads]
    return int(torch.stack(bad).sum())


def flag_report(
    flags: list[tuple[str, Tensor | None]], limit: int = LIMIT
) -> list[dict]:
    """In-backward steps: the first `limit` parameters whose found_inf flag was set (their
    gradients were consumed in backward, so NaN / Inf counts are not available); a
    None flag (an update without one) names nothing."""
    flags = [(name, f) for name, f in flags if f is not None]
    if not flags:
        return []
    dev = flags[0][1].device
    values = torch.stack([f.reshape(()).float().to(dev) for _, f in flags]).tolist()
    out = [
        {"param": name, "kind": kind_of(name), "nan": None, "inf": None}
        for (name, _), v in zip(flags, values)
        if v
    ]
    return out[:limit]
