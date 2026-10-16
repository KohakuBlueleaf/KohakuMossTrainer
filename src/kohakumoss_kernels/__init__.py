"""Kernel library for KohakuMoss: ops, backends, constraints and tuning."""

from kohakumoss_kernels import registry
from kohakumoss_kernels.hardware.arch import Arch, arch_of
from kohakumoss_kernels.registry import available, register, resolve, set_priority

__all__ = [
    "Arch",
    "arch_of",
    "available",
    "register",
    "registry",
    "resolve",
    "set_priority",
]

registry.load_backends()
