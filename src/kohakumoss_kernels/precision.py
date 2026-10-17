"""Precision bar: a result against an fp64 reference, relative to PyTorch (the eager
composition) run in the same dtype. See docs/internals/fp16.md#the-spec-and-the-bar.

A tensor fails when it is non-finite, or when BOTH its absolute error (max |out - ref|)
and its relative error (max |out - ref| / |ref| over nonzero ref) exceed `FACTOR` times
the eager composition's.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

FACTOR = 2.0


@dataclass(frozen=True)
class Error:
    """Absolute and relative error of one tensor against its reference."""

    abs: float
    rel: float


def error(out: Tensor, ref: Tensor) -> Error:
    """max |out - ref| and max |out - ref| / |ref| over the nonzero elements of ref."""
    d = (out.detach().double() - ref.detach().double()).abs()
    r = ref.detach().double().abs()
    nz = r > 0
    rel = (d[nz] / r[nz]).max().item() if bool(nz.any()) else 0.0
    return Error(d.max().item() if d.numel() else 0.0, rel)


@dataclass(frozen=True)
class Verdict:
    """The bar applied to one tensor: its error, the eager error, finiteness."""

    out: Error
    eager: Error
    finite: bool
    factor: float = FACTOR

    @property
    def ok(self) -> bool:
        worse_abs = self.out.abs > self.factor * self.eager.abs
        worse_rel = self.out.rel > self.factor * self.eager.rel
        return self.finite and not (worse_abs and worse_rel)


def judge(out: Tensor, eager: Tensor, ref64: Tensor, factor: float = FACTOR) -> Verdict:
    """`out` and the same-dtype `eager` result, both against the fp64 `ref64`."""
    finite = bool(torch.isfinite(out).all())
    return Verdict(error(out, ref64), error(eager, ref64), finite, factor)
