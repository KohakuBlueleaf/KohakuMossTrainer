"""Shared by Triton ops: launching inside `triton_op` traces and dtype mapping."""

import torch
from torch.library import wrap_triton

_PLAIN = (torch.Tensor, torch.nn.Parameter)


class Launcher:
    """A `JITFunction` launched directly on real tensors and through `wrap_triton`
    while a `triton_op` is traced (fake / functional tensors)."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self._launch(grid, args, kwargs)

    def _launch(self, grid, args, kwargs):
        first = next(a for a in args if isinstance(a, torch.Tensor))
        if type(first) in _PLAIN:
            return self.fn[grid](*args, **kwargs)
        return wrap_triton(self.fn)[grid](*args, **kwargs)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pow2_bucket(n: int, floor: int = 64) -> int:
    """Smallest power of two >= max(n, floor): the tuning key of a length."""
    b = floor
    while b < n:
        b *= 2
    return b
