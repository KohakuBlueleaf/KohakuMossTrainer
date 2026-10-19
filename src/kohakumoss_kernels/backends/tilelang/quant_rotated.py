"""TileLang rotated int8 linear (ConvRot, the sm70 / sm75 path): the registry's
`group_hadamard` for the device (the Triton butterflies or the compiled reference,
chosen per shape), then the plain TileLang int8 kernels (`quant.py`), one pass each
way. See docs/internals/quant-kernels.md#rotated-int8-linear."""

from torch import Tensor

from kohakumoss_kernels import registry
from kohakumoss_kernels.backends.tilelang import quant as plain
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.quant.rotated import (
    compose,
    group_supported,
    make_quant_linear_rotated,
    register_rotated_launchers,
)

_ROTATIONS: dict[tuple, registry.Bound] = {}


def rotate(t: Tensor, group: int) -> Tensor:
    """`group_hadamard` of [R, K] `t`, the op resolved once per (device, dtype)."""
    key = (t.device, t.dtype)
    op = _ROTATIONS.get(key)
    if op is None:
        op = _ROTATIONS[key] = registry.resolve("group_hadamard", t.device, t.dtype)
    return op(t, group)


def _supported(query):
    fmt = query.get("fmt", "int8")
    if fmt != "int8":
        return f"tilelang quant serves int8 only, not {fmt}"
    return group_supported(query)


register_rotated_launchers("tilelang", *compose(plain.fwd, plain.dgrad, rotate))
registry.register(
    "quant_linear_rotated",
    "tilelang",
    make_quant_linear_rotated("tilelang"),
    supports=requires(kinds=("cuda",), min_sm=70, extra=_supported),
)
