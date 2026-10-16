"""Eager reference backend: the spec of every op, and the CPU path."""

from kohakumoss_kernels import quant  # noqa: F401  (registers the eager quant_linear)
from kohakumoss_kernels.backends.eager import attention, ffn, loss, norm, rope
from kohakumoss_kernels.registry import register

for _module in (attention, ffn, loss, norm, rope):
    for _name in _module.__all__:
        register(_name, "eager", getattr(_module, _name))
