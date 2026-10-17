"""Precision checks against an fp64 reference, judged in ULP of the tested dtype.

The bar: a backend's error vs fp64 must not exceed `factor` times the error of the eager
composition run in the same dtype, in both the elementwise-max and RMS metrics. There is
no absolute ULP floor by default (`floor_ulp=0`).
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def ulp(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).eps


def ulp_error(out: Tensor, ref: Tensor, dtype: torch.dtype, mode: str) -> float:
    """Error in ULP of `dtype`: `elementwise` = max |d| / max(|ref|, tiny); `rms` = rms(d)/rms(ref)."""
    d = out.double() - ref.double()
    r = ref.double()
    match mode:
        case "elementwise":
            scale = r.abs().clamp_min(r.abs().max().item() * 1e-3 + 1e-30)
            return (d.abs() / scale).max().item() / ulp(dtype)
        case "rms":
            denom = r.square().mean().sqrt().item() + 1e-30
            return d.square().mean().sqrt().item() / denom / ulp(dtype)
        case _:
            raise ValueError(mode)


@dataclass
class BarResult:
    backend_elem: float
    backend_rms: float
    eager_elem: float
    eager_rms: float

    def ok(
        self, factor: float, floor_ulp: float, factor_elem: float | None = None
    ) -> bool:
        fe = factor if factor_elem is None else factor_elem
        return self.backend_elem <= max(
            fe * self.eager_elem, floor_ulp
        ) and self.backend_rms <= max(factor * self.eager_rms, floor_ulp)


def bar(out: Tensor, eager: Tensor, ref64: Tensor, dtype: torch.dtype) -> BarResult:
    return BarResult(
        ulp_error(out, ref64, dtype, "elementwise"),
        ulp_error(out, ref64, dtype, "rms"),
        ulp_error(eager, ref64, dtype, "elementwise"),
        ulp_error(eager, ref64, dtype, "rms"),
    )


def assert_bar(
    out: Tensor,
    eager: Tensor,
    ref64: Tensor,
    dtype: torch.dtype,
    factor: float = 1.1,
    floor_ulp: float = 0.0,
    name: str = "",
    factor_elem: float | None = None,
) -> None:
    """Assert `out` is within the eager-in-dtype precision bar against `ref64`.

    `factor` bounds the RMS metric; `factor_elem` (default `factor`) bounds the
    elementwise max, which near-zero outputs dominate in reductions.
    """
    assert torch.isfinite(out).all(), f"{name}: non-finite output"
    assert ref64.abs().max() > 0, f"{name}: reference is all zero"
    res = bar(out, eager, ref64, dtype)
    assert res.ok(factor, floor_ulp, factor_elem), f"{name}: {res}"
