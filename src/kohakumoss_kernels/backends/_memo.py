"""Values built from a live tensor (segment bounds, tile plans), reused while that tensor
object is alive at the same in-place version and dropped with it; an inference tensor
is rebuilt every call."""

import weakref
from collections.abc import Callable

from torch import Tensor


def version(t: Tensor) -> int:
    """`t`'s in-place version counter; inference tensors carry none (0)."""
    return 0 if t.is_inference() else t._version


class TensorMemo:
    """`get(t, key, build)`: `build()` once per (tensor `t`, `key`) and version of `t`;
    every call for an inference tensor."""

    def __init__(self):
        self.rows: dict[tuple, tuple] = {}

    def get(self, t: Tensor, key, build: Callable[[], object]):
        if t.is_inference():
            return build()
        ident = (id(t), key)
        at = version(t)
        hit = self.rows.get(ident)
        if hit is not None and hit[0]() is t and hit[1] == at:
            return hit[2]
        value = build()
        if hit is None or hit[0]() is not t:
            weakref.finalize(t, self.rows.pop, ident, None)
        self.rows[ident] = (weakref.ref(t), at, value)
        return value
