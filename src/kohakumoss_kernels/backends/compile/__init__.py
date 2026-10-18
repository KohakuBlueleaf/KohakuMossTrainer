"""`compile` backend: torch.compile of each op's eager reference (and of the eager
compositions of the fused linear ops), compiled once per op with dynamic shapes.

Registered for every block op; kernel backends also choose against it per shape
(`choice.py`). Its probe on a GPU is Triton's; on CPU it needs the host C++ compiler.
See docs/internals/kernel-dev.md#compile-backend.
"""

import torch

from kohakumoss_kernels import status
from kohakumoss_kernels.backends.compile.ops import COMPILED, compiled_op
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain, installed
from kohakumoss_kernels.registry import register

__all__ = ["COMPILED", "compiled_op"]


def check(device: torch.device) -> None:
    """Raise when inductor cannot generate code for `device` here."""
    if device.type == "cpu":
        toolchain = ensure_toolchain()
        if not toolchain.ok:
            raise RuntimeError(toolchain.detail)
        return
    if not installed("triton"):
        raise RuntimeError(f"inductor on {device.type} needs triton")
    why = status.probe("triton", device)
    if why is not None:
        raise RuntimeError(why)


status.register_check("compile", check)
for _name in COMPILED:
    register(_name, "compile", compiled_op(_name))
