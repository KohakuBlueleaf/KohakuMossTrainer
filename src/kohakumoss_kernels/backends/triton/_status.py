"""Triton backend smoke check: build and launch one minimal kernel on a device."""

import torch
import triton
import triton.language as tl

from kohakumoss_kernels.backends.triton._common import Launcher
from kohakumoss_kernels.hardware.current import is_accel


@triton.jit
def _double_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)


_DOUBLE = Launcher(_double_kernel)


def check(device: torch.device) -> None:
    """Raise when a Triton kernel cannot be built, launched or give the right result."""
    if not is_accel(device):
        raise RuntimeError(f"triton does not run on {device.type}")
    x = torch.ones(16, device=device)
    _DOUBLE[(1,)](x, 16, BLOCK=16)
    if not bool((x == 2).all()):
        raise RuntimeError(f"triton smoke kernel gave a wrong result on {device}")
