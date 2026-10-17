"""`compile` backend: torch.compile of each op's eager spec (and of the eager
compositions of the fused linear ops), compiled once per op with dynamic shapes.

Registered for every block op so the registry can select it; kernel backends also race
against it per shape (`race.py`). Its health on a GPU is Triton's (inductor's GPU code is
Triton); on CPU it needs the host C++ compiler.
"""

import torch

from kohakumoss_kernels import health
from kohakumoss_kernels.backends.compile.ops import COMPILED, compiled_op
from kohakumoss_kernels.hardware.toolchain import ensure_toolchain, installed
from kohakumoss_kernels.registry import register

__all__ = ["COMPILED", "compiled_op"]

TOOLCHAIN = ensure_toolchain()


def check(device: torch.device) -> None:
    """Raise when inductor cannot generate code for `device` here."""
    if device.type == "cpu":
        if not TOOLCHAIN.ok:
            raise RuntimeError(TOOLCHAIN.detail)
        return
    if not installed("triton"):
        raise RuntimeError(f"inductor on {device.type} needs triton")
    why = health.healthy("triton", device)
    if why is not None:
        raise RuntimeError(why)


health.register_check("compile", check)
for _name in COMPILED:
    register(_name, "compile", compiled_op(_name))
