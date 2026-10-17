"""Precision assertion over `kohakumoss_kernels.precision` (fp64 reference, eager bar)."""

from torch import Tensor

from kohakumoss_kernels.precision import FACTOR, Error, Verdict, error, judge

__all__ = ["FACTOR", "Error", "Verdict", "assert_bar", "error", "judge"]


def assert_bar(out: Tensor, eager: Tensor, ref64: Tensor, name: str = "") -> None:
    """Assert `out` is finite and not worse than `FACTOR` x the same-dtype `eager`
    result in both absolute and relative error against `ref64`."""
    assert ref64.abs().max() > 0, f"{name}: reference is all zero"
    v = judge(out, eager, ref64)
    assert v.ok, f"{name}: {v}"
