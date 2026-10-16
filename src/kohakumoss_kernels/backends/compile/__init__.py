"""`compile` backend: torch.compile of each op's eager spec (and of the eager
compositions of the fused linear ops), compiled once per op with dynamic shapes.

Registered for every block op so the registry can select it; kernel backends also race
against it per shape through `race.choose` and fall back to it where it wins.
"""

from kohakumoss_kernels.backends.compile.ops import COMPILED, compiled_op
from kohakumoss_kernels.registry import register

__all__ = ["COMPILED", "compiled_op"]

for _name in COMPILED:
    register(_name, "compile", compiled_op(_name))
