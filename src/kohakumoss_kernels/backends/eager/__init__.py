"""Eager reference backend: what every op computes, and the CPU path. Every name in a
module's `__all__` is registered as that op."""

from kohakumoss_kernels import fp8, int8, quant  # noqa: F401  (the eager quant ops)
from kohakumoss_kernels.backends.eager import (
    attention,
    ffn,
    hadamard,
    loss,
    norm,
    rope,
    textenc,
)
from kohakumoss_kernels.quant import rotated  # noqa: F401  (the eager rotated op)
from kohakumoss_kernels.registry import register

for _module in (attention, ffn, hadamard, loss, norm, rope, textenc):
    for _name in _module.__all__:
        register(_name, "eager", getattr(_module, _name))
