"""Forward + backward error-statistics check of one op against its eager reference.

The same rounded inputs run through (a) the backend in the test dtype, (b) the eager
reference in the test dtype and (c) the eager reference in fp64; outputs and every leaf
gradient of (a) must have no regression (tests/_util/error_stats.py) against (c),
relative to (b). (a)'s output dtype is (b)'s or the promotion of the operand dtypes
(a kernel never forces a 16-bit output). The incoming gradient is drawn on the output
dtype's grid.
"""

from collections.abc import Callable, Sequence
from functools import reduce

import torch
from torch import Tensor

from tests._util.error_stats import assert_no_regression


def _leaf(t, dtype=None, grad=True):
    """A fresh leaf with `t`'s values, shape and strides (strided views stay strided)."""
    if not isinstance(t, Tensor):
        return t
    out = torch.empty_strided(
        t.shape, t.stride(), dtype=dtype or t.dtype, device=t.device
    )
    out.copy_(t.detach())
    if grad and out.is_floating_point():
        out.requires_grad_(True)
    return out


def _as_tuple(out) -> tuple[Tensor, ...]:
    return out if isinstance(out, tuple) else (out,)


def operand_dtype(args: Sequence) -> torch.dtype | None:
    """The promotion of the floating operand dtypes: the output dtype a kernel storing
    in its operands' natural dtype has."""
    dtypes = [a.dtype for a in args if isinstance(a, Tensor) and a.is_floating_point()]
    return reduce(torch.promote_types, dtypes) if dtypes else None


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
    backward: bool = True,
) -> None:
    """Assert `backend` has no regression on outputs and the gradients of `grads` leaves;
    `dtype` is the tested activation dtype (the gradients keep their leaves' dtypes)."""
    lb, ob = run(backend, args, grads, False)
    le, oe = run(eager, args, grads, False)
    lr, orf = run(eager, args, grads, True)
    natural = operand_dtype(args)
    for i, (a, b, c) in enumerate(zip(ob, oe, orf)):
        assert a.dtype in (b.dtype, natural), (
            f"{name} out{i}: dtype {a.dtype} is neither eager's {b.dtype} nor the "
            f"operands' {natural}"
        )
        assert a.shape == b.shape, f"{name} out{i}: shape {a.shape} != {b.shape}"
        assert_no_regression(
            a.detach(), b.detach(), c.detach(), f"{name} {dtype} out{i}"
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
        assert_no_regression(gb, ge, gr, f"{name} {dtype} grad{i}")
