"""Eager reference backend: what every op computes, and the CPU path. Every name in a
module's `__all__` is registered as that op."""

from kohakumoss_kernels import fp8, quant  # noqa: F401  (register the eager quant ops)
from kohakumoss_kernels.backends.eager import attention, ffn, loss, norm, rope, textenc
from kohakumoss_kernels.registry import register

for _module in (attention, ffn, loss, norm, rope, textenc):
    for _name in _module.__all__:
        register(_name, "eager", getattr(_module, _name))
