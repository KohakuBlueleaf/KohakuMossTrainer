"""The fused-vs-eager bar: a fused result fails only when its error against the fp64
reference is clearly worse than the eager body's in both senses at once, absolute (RMS
of the difference) and relative (RMS of the per-element relative difference)."""

from torch import Tensor

WORSE = 2.0
FLOOR = 1e-2


def abs_err(out: Tensor, ref: Tensor) -> float:
    """RMS of out - ref."""
    return (out.double() - ref).square().mean().sqrt().item()


def rel_err(out: Tensor, ref: Tensor) -> float:
    """RMS of |out - ref| / (|ref| + FLOOR * mean |ref|): per-element relative error,
    with a fraction of the tensor's own scale keeping zeros from dividing by ~0."""
    ref = ref.double()
    denom = ref.abs() + FLOOR * ref.abs().mean()
    return ((out.double() - ref).abs() / denom).square().mean().sqrt().item()


def errors(out: Tensor, eager: Tensor, ref: Tensor) -> dict[str, float]:
    return {
        "abs": abs_err(out, ref),
        "abs_eager": abs_err(eager, ref),
        "rel": rel_err(out, ref),
        "rel_eager": rel_err(eager, ref),
    }


def explodes(out: Tensor, eager: Tensor, ref: Tensor, worse: float = WORSE) -> bool:
    """`out` is clearly worse than `eager` (more than `worse` x) in abs AND rel."""
    e = errors(out, eager, ref)
    return e["abs"] > worse * e["abs_eager"] and e["rel"] > worse * e["rel_eager"]


def check(out: Tensor, eager: Tensor, ref: Tensor, what: str) -> None:
    e = errors(out, eager, ref)
    assert not explodes(out, eager, ref), (
        f"{what}: fused abs {e['abs']:.3e} vs eager {e['abs_eager']:.3e}, "
        f"rel {e['rel']:.3e} vs eager {e['rel_eager']:.3e} (both > {WORSE}x)"
    )
