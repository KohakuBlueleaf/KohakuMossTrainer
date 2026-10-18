"""Byte units: `GIB` / `MIB` and their conversions (every memory figure is binary)."""

MIB = 2**20
GIB = 2**30


def gib(n: int | float) -> float:
    """`n` bytes in GiB."""
    return n / GIB


def mib(n: int | float) -> float:
    """`n` bytes in MiB."""
    return n / MIB
