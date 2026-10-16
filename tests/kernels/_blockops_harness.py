"""Forward + backward precision check of one op against its eager spec.

The same rounded inputs run through (a) the backend in the test dtype, (b) the eager
spec in the test dtype and (c) the eager spec in fp64; outputs and every leaf gradient
of (a) must meet the precision bar of tests/_util/precision.py against (c), relative
to (b). The incoming gradient is drawn on the output dtype's grid.
"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from tests._util.precision import assert_bar


def _leaf(t, dtype=None, grad=True):
    if not isinstance(t, Tensor):
        return t
    out = t.detach().clone() if dtype is None else t.detach().to(dtype)
    if grad and out.is_floating_point():
        out.requires_grad_(True)
    return out


def _as_tuple(out) -> tuple[Tensor, ...]:
    return out if isinstance(out, tuple) else (out,)


def run(fn: Callable, args: Sequence, grads: Sequence[bool], dtype64: bool):
    leaves = [
        _leaf(
            a,
            (
                torch.float64
                if dtype64 and isinstance(a, Tensor) and a.is_floating_point()
                else None
            ),
            g,
        )
        for a, g in zip(args, grads)
    ]
    outs = _as_tuple(fn(*leaves))
    return leaves, outs


def check(
    backend: Callable,
    eager: Callable,
    args: Sequence,
    grads: Sequence[bool],
    dtype: torch.dtype,
    name: str,
    seed: int = 1,
    factor: float = 1.1,
    floor_ulp: float = 2.0,
    backward: bool = True,
) -> None:
    """Assert `backend` meets the bar on outputs and the gradients of `grads` leaves."""
    lb, ob = run(backend, args, grads, False)
    le, oe = run(eager, args, grads, False)
    lr, orf = run(eager, args, grads, True)
    for i, (a, b, c) in enumerate(zip(ob, oe, orf)):
        assert a.dtype == b.dtype, f"{name} out{i}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{name} out{i}: shape {a.shape} != {b.shape}"
        assert_bar(
            a.detach(),
            b.detach(),
            c.detach(),
            dtype,
            factor,
            floor_ulp,
            f"{name} out{i}",
        )
    if not backward:
        return
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gouts = [torch.randn(o.shape, generator=gen).to(o.device, o.dtype) for o in ob]
    for leaves, outs in ((lb, ob), (le, oe), (lr, orf)):
        total = sum((o * g.to(o.dtype)).sum() for o, g in zip(outs, gouts))
        total.backward()
    for i, need in enumerate(grads):
        if not need or not isinstance(args[i], Tensor):
            continue
        gb, ge, gr = lb[i].grad, le[i].grad, lr[i].grad
        assert gb is not None, f"{name} grad{i}: missing"
        assert gb.dtype == args[i].dtype, f"{name} grad{i}: dtype {gb.dtype}"
        assert_bar(gb, ge, gr, dtype, factor, floor_ulp, f"{name} grad{i}")
