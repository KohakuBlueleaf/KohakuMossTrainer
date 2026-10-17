"""Per-layer binding to LyCORIS's measured, precision-gated tier race: a `TierSlot`
asks `lycoris.kernels.tier.best` once per shape bucket and keeps the arm LyCORIS
settled on (see docs/internals/adapter-binding.md)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
from lycoris.kernels import tier
from lycoris.kernels.plans import tune
from lycoris.kernels.select import FUSED, call_compiled
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


def _compiled_body(ref, ref_args, bypass: bool, ops: tuple) -> Tensor:
    """The compiled body; inside an enclosing torch.compile, the body itself. A
    bypass op's activation (ops[0]) has its token axes left dynamic."""
    args = ref_args(ops)
    if torch.compiler.is_compiling():
        return ref(*args)
    return call_compiled(ref, *args, token_arg=ops[0] if bypass else None)


def body(op: Op, pick: str) -> Callable[[tuple], Tensor]:
    """The function of the operand tuple that a tier pick runs."""
    if pick in FUSED:
        return partial(op.fused, pick)
    if pick in op.extras:
        return op.extras[pick]
    if pick == "compile":
        return partial(_compiled_body, op.ref, op.ref_args, "bypass" in op.name)
    return partial(_torch_body, op.ref, op.ref_args)


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
    return tier.is_settled(name, tensors, supported)


def token_bucket(t: Tensor) -> tuple:
    """Bucket of a (tokens, channels) activation: LyCORIS's token bucket of the
    row count, dtype and grad pattern."""
    grad = torch.is_grad_enabled()
    return (tune.bucket_tokens(t.shape[0]), t.dtype, t.requires_grad, grad)


def static_bucket(t: Tensor) -> tuple:
    """Bucket of a fixed-shape operand (a base weight): dtype and grad pattern."""
    return (t.dtype, t.requires_grad, torch.is_grad_enabled())


TOKEN_PATTERNS = ((True, True), (False, True), (False, False))
WEIGHT_PATTERNS = ((False, True), (True, True), (False, False))


def _warm_one(slot, t: Tensor, grad: bool) -> int:
    with torch.set_grad_enabled(grad):
        y = slot(t)
        if y.requires_grad:
            y.float().sum().backward()
    return 1


class TierSlot:
    """Callable(t) -> `op` through the tier LyCORIS picked for t's bucket.

    `width` is the channel count of a (tokens, channels) operand; `shape` the shape of
    a fixed operand (a base or merged weight, fp32). `warm` calls the slot on scratch
    operands of each bucket and grad pattern.
    """

    def __init__(
        self,
        op: Op,
        bucket: Callable[[Tensor], tuple],
        width: int | None = None,
        shape: tuple | None = None,
    ):
        self.op = op
        self.bucket = bucket
        self.width = width
        self.shape = shape
        self.arms: dict[tuple, Callable[[Tensor], Tensor]] = {}

    def warm(self, rows: list[int], dtype: torch.dtype, device) -> int:
        """Resolve every bucket of `rows` (or the fixed operand) on scratch inputs."""
        calls = 0
        if self.width is not None:
            for n in rows:
                for needs_grad, grad in TOKEN_PATTERNS:
                    x = torch.randn(n, self.width, device=device, dtype=dtype)
                    calls += _warm_one(self, x.requires_grad_(needs_grad), grad)
            return calls
        for needs_grad, grad in WEIGHT_PATTERNS:
            w = torch.randn(self.shape, device=device, dtype=torch.float32)
            calls += _warm_one(self, w.requires_grad_(needs_grad), grad)
        return calls

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
