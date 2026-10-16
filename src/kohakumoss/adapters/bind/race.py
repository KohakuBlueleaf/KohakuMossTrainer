"""Per-layer binding to LyCORIS's measured, precision-gated tier race.

A `TierSlot` holds one layer's `Op`. On the first call of a shape bucket it asks
`lycoris.kernels.tier.best` with the real operands (racing the fused backends, the
compile tier, the eager body and the op's exact extras on a miss of LyCORIS's table)
and keeps the picked arm; every later call of that bucket is one dict lookup and the
bound arm. A pick LyCORIS holds provisionally (an all-zero reference, e.g. freshly
zero-initialised factors) is not kept: the slot asks again on every call, which is
LyCORIS's own re-race schedule, until LyCORIS records a decision.

The bucket covers what varies per call (token bucket, activation dtype, grad
pattern); factor shapes, dtypes and grad flags are fixed between `AdapterSet.bind()`
calls, each of which builds fresh slots.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from lycoris.kernels import tier
from lycoris.kernels.plans import tune
from lycoris.kernels.select import FUSED, call_compiled, choose, device_of
from torch import Tensor


@dataclass(frozen=True)
class Op:
    """One LyCORIS tier op over a layer.

    `operands(t)` is the op's operand tuple for runtime tensor `t` (the tensors
    LyCORIS keys and races on); every arm reads it: `fused(backend, ops)`,
    `ref(*ref_args(ops))` and each `extras[name](ops)`.
    """

    name: str
    operands: Callable[[Tensor], tuple]
    supported: bool
    fused: Callable[[str, tuple], Tensor]
    ref: Callable
    ref_args: Callable[[tuple], tuple]
    extras: dict = field(default_factory=dict)


def _torch_body(ref, ref_args, ops: tuple) -> Tensor:
    return ref(*ref_args(ops))


def _compiled_body(ref, ref_args, ops: tuple) -> Tensor:
    """The compiled body; inside an enclosing torch.compile, the body itself."""
    args = ref_args(ops)
    if torch.compiler.is_compiling():
        return ref(*args)
    return call_compiled(ref, *args)


def body(op: Op, pick: str) -> Callable[[tuple], Tensor]:
    """The function of the operand tuple that a tier pick runs."""
    if pick in FUSED:
        return partial(op.fused, pick)
    if pick in op.extras:
        return op.extras[pick]
    fn = _compiled_body if pick == "compile" else _torch_body
    return partial(fn, op.ref, op.ref_args)


def _on(operands, fn, t: Tensor) -> Tensor:
    return fn(operands(t))


def arm(op: Op, pick: str) -> Callable[[Tensor], Tensor]:
    """The bound callable of the runtime tensor for a tier pick."""
    return partial(_on, op.operands, body(op, pick))


def _fused_on(op: Op, ops: tuple, backend: str) -> Tensor:
    return op.fused(backend, ops)


def race_args(op: Op, t: Tensor) -> tuple:
    """The positional arguments of `tier.best` / `tier.run` for runtime tensor `t`;
    every argument shares one operand tuple."""
    ops = op.operands(t)
    extras = {name: partial(fn, ops) for name, fn in op.extras.items()}
    return (
        op.name,
        ops,
        op.supported,
        None,
        partial(_fused_on, op, ops),
        op.ref,
        op.ref_args(ops),
        extras,
    )


def settled(name: str, tensors: tuple, supported: bool) -> bool:
    """LyCORIS keeps its pick for these operands for good (or has nothing to race)."""
    if choose(tensors, supported, None) not in FUSED or not tune.tuning_enabled():
        return True
    with torch.cuda.device(device_of(tensors)):
        return tune.lookup(tier._key(name, tensors)) is not None


def token_bucket(t: Tensor) -> tuple:
    """Bucket of a (tokens, channels) activation: LyCORIS's token bucket of the
    row count, dtype and grad pattern."""
    grad = torch.is_grad_enabled()
    return (tune.bucket_tokens(t.shape[0]), t.dtype, t.requires_grad, grad)


def static_bucket(t: Tensor) -> tuple:
    """Bucket of a fixed-shape operand (a base weight): dtype and grad pattern."""
    return (t.dtype, t.requires_grad, torch.is_grad_enabled())


class TierSlot:
    """Callable(t) -> `op` through the tier LyCORIS picked for t's bucket."""

    def __init__(self, op: Op, bucket: Callable[[Tensor], tuple]):
        self.op = op
        self.bucket = bucket
        self.arms: dict[tuple, Callable[[Tensor], Tensor]] = {}

    def __call__(self, t: Tensor) -> Tensor:
        run = self.arms.get(self.bucket(t))
        if run is None:
            run = self.resolve(t)
        return run(t)

    def resolve(self, t: Tensor) -> Callable[[Tensor], Tensor]:
        """Ask the race for t's bucket; keep the arm once LyCORIS has settled it."""
        args = race_args(self.op, t)
        run = arm(self.op, tier.best(*args))
        if settled(*args[:3]) and not torch.compiler.is_compiling():
            self.arms[self.bucket(t)] = run
        return run


def flat(fn: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """`fn` on x viewed as (tokens, channels), reshaped back to x's leading axes."""
    y = fn(x.reshape(-1, x.shape[-1]))
    return y.reshape(*x.shape[:-1], y.shape[-1])
