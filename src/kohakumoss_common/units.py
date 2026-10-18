"""Byte units: `GIB` / `MIB` and the GiB conversion (every memory figure is binary)."""

MIB = 2**20
GIB = 2**30


def gib(n: int | float) -> float:
    """`n` bytes in GiB."""
    return n / GIB
