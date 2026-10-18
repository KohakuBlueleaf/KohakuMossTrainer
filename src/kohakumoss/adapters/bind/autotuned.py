"""AutotunedOp: one adapter op bound, per problem bucket, to the faster of its two
candidates on this card: the LyCORIS fused kernel of the layer's backend (`kernel`) or
the body candidate (`compile`, the compiled eager body; `eager` for a body through a
custom autograd.Function). The kernel tuner measures each problem once and stores the
winner per card (see docs/internals/adapter-binding.md#autotuned-ops)."""

import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache, partial

import torch
import torch._dynamo
from lycoris.kernels.select import fusable
from torch import Tensor

from kohakumoss.adapters.bind import bodies
from kohakumoss_common.units import MIB
from kohakumoss_kernels.backends.compile.ops import CompiledOp
from kohakumoss_kernels.dtypes import short
from kohakumoss_kernels.errors import BUILD_ERRORS
from kohakumoss_kernels.hardware.current import rng_kept, saved_tensor_hooks_active
from kohakumoss_kernels.hardware.memory import release
from kohakumoss_kernels.tuning.buckets import TOKEN_FLOOR, pow2, pow2_buckets
from kohakumoss_kernels.tuning.tuner import (
    OverBound,
    decided,
    inherit,
    on_reset,
    row,
    select,
    stored,
)

ERRORS = (*BUILD_ERRORS, ValueError)
DYNAMIC = {"tokens": None, "static": False, "dynamic": True}
TOKEN_PATTERNS = ((True, True), (False, True), (False, False))
WEIGHT_PATTERNS = ((False, True), (True, True), (False, False))
PAIR_PATTERNS = ((True, True), (False, False))
BODY = 1  # index of the body candidate in `AutotunedOp.configs`
_OPS: "weakref.WeakSet[AutotunedOp]" = weakref.WeakSet()


def token_buckets(batch_max: int) -> list[int]:
    """One row count per token bucket up to `batch_max`'s (powers of two from
    TOKEN_FLOOR): the bucket's top, or `batch_max` when that is smaller."""
    return pow2_buckets(batch_max)


def _randn(shape, dtype, device, needs_grad: bool) -> Tensor:
    t = torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
    return t.requires_grad_(needs_grad)


def _built(problems: Iterator[tuple], dtype, device) -> Iterator[tuple]:
    """(runtime input, grad mode) per problem, each built when it is reached."""
    for build, grad in problems:
        yield build(dtype, device), grad


class Tokens:
    """Runtime input: a (tokens, `width`) activation."""

    def __init__(self, width: int):
        self.width = width

    def bucket(self, x: Tensor) -> tuple:
        grad = torch.is_grad_enabled()
        return (pow2(x.shape[0], TOKEN_FLOOR), x.dtype, x.requires_grad, grad)

    def problems(self, rows) -> Iterator[tuple]:
        """(build(dtype, device) -> runtime input, grad mode) per bucket and pattern."""
        for n in rows:
            for needs_grad, grad in TOKEN_PATTERNS:
                yield partial(self._build, n, needs_grad), grad

    def _build(self, n: int, needs_grad: bool, dtype, device) -> Tensor:
        return _randn((n, self.width), dtype, device, needs_grad)

    def examples(self, rows, dtype, device) -> Iterator[tuple]:
        return _built(self.problems(rows), dtype, device)


class TokensWithBase:
    """Runtime input: the pair (activation (tokens, `width`), base output (tokens,
    `base_width`))."""

    def __init__(self, width: int, base_width: int):
        self.width = width
        self.base_width = base_width

    def bucket(self, t: tuple) -> tuple:
        x, base = t
        grad = torch.is_grad_enabled()
        n = pow2(x.shape[0], TOKEN_FLOOR)
        return (n, x.dtype, x.requires_grad, base.dtype, base.requires_grad, grad)

    def problems(self, rows) -> Iterator[tuple]:
        """(build(dtype, device) -> runtime input, grad mode) per bucket and pattern."""
        for n in rows:
            for needs_grad, grad in TOKEN_PATTERNS:
                yield partial(self._build, n, needs_grad), grad

    def _build(self, n: int, needs_grad: bool, dtype, device) -> tuple:
        x = _randn((n, self.width), dtype, device, needs_grad)
        return x, _randn((n, self.base_width), dtype, device, needs_grad)

    def examples(self, rows, dtype, device) -> Iterator[tuple]:
        return _built(self.problems(rows), dtype, device)


class Weight:
    """Runtime input: a weight of `shape` (the fp32 base or merged weight)."""

    def __init__(self, shape: tuple):
        self.shape = tuple(shape)

    def bucket(self, w: Tensor) -> tuple:
        return (w.dtype, w.requires_grad, torch.is_grad_enabled())

    def problems(self, rows) -> Iterator[tuple]:
        """(build(dtype, device) -> runtime input, grad mode) per pattern."""
        for needs_grad, grad in WEIGHT_PATTERNS:
            yield partial(self._build, needs_grad), grad

    def _build(self, needs_grad: bool, dtype, device) -> Tensor:
        return _randn(self.shape, torch.float32, device, needs_grad)

    def examples(self, rows, dtype, device) -> Iterator[tuple]:
        return _built(self.problems(rows), dtype, device)


class WeightWithDelta:
    """Runtime input: the pair (base weight, delta), both of `shape` (fp32)."""

    def __init__(self, shape: tuple):
        self.shape = tuple(shape)

    def bucket(self, t: tuple) -> tuple:
        w, dw = t
        grad = torch.is_grad_enabled()
        return (w.dtype, w.requires_grad, dw.dtype, dw.requires_grad, grad)

    def problems(self, rows) -> Iterator[tuple]:
        """(build(dtype, device) -> runtime input, grad mode) per pattern."""
        for needs_grad, grad in PAIR_PATTERNS:
            yield partial(self._build, needs_grad), grad

    def _build(self, needs_grad: bool, dtype, device) -> tuple:
        w = _randn(self.shape, torch.float32, device, False)
        return w, _randn(self.shape, torch.float32, device, needs_grad)

    def examples(self, rows, dtype, device) -> Iterator[tuple]:
        return _built(self.problems(rows), dtype, device)


def _always(backend: str) -> bool:
    return True


@dataclass(frozen=True)
class Op:
    """One adapter op over a layer.

    `operands(t)` is the operand tuple for runtime input `t` (shaped per `runtime`);
    `kernel(backend, ops)` the LyCORIS fused op; `body(ops, *consts)` the eager formula
    (lycoris.functional on backend "torch"), compiled for the `compile` candidate;
    `in_scope(backend)` whether that backend's fused kernel covers this layer; `tokens`
    the operand indices that carry the token axis.
    """

    name: str
    runtime: object
    operands: Callable
    kernel: Callable[[str, tuple], Tensor]
    body: Callable
    consts: tuple = ()
    in_scope: Callable[[str], bool] = _always
    tokens: tuple[int, ...] = ()


def eager(op: Op) -> Callable[[tuple], Tensor]:
    """The eager candidate: `op.body` on the operand tuple."""
    body, consts = op.body, op.consts

    def run(ops: tuple) -> Tensor:
        return body(ops, *consts)

    return run


@cache
def _graph(
    body: Callable, dynamic: bool | None, shapes: tuple, backend="inductor"
) -> Callable:
    return CompiledOp(f"adapter_{body.__name__}", dynamic, backend, eager=body)


def fixed_shapes(ops: tuple, marked: frozenset) -> tuple:
    """Every operand's shape without the token axis of the `marked` operands (None
    for a missing operand): what the compile cache of a call is keyed on."""
    return tuple(
        (
            None
            if not isinstance(a, Tensor)
            else tuple(a.shape[1:]) if i in marked else tuple(a.shape)
        )
        for i, a in enumerate(ops)
    )


class CompiledBody:
    """The compile candidate: torch.compile of `op.body`, one `CompiledOp` per body and
    operand shapes apart from the token axes, so each Dynamo cache holds one graph
    whatever ran before it: a checkpointed forward and its recompute run graphs traced
    from the same inputs (`dynamic`: "tokens" marks the token axes dynamic, "static",
    "dynamic"); inside an enclosing torch.compile, the body itself. Callable on any
    operand tuple; `at(ops)` is the same bound to the `CompiledOp` of `ops`' shapes.
    See docs/internals/adapter-binding.md#checkpoint-recompute."""

    def __init__(self, op: Op, dynamic: str):
        self.body, self.consts, self.mode = op.body, op.consts, DYNAMIC[dynamic]
        self.marked = op.tokens if dynamic == "tokens" else ()
        self.axes: dict[int, frozenset] = {}

    def _axes(self, arity: int) -> frozenset:
        axes = self.axes.get(arity)
        if axes is None:
            axes = self.axes[arity] = frozenset(i % arity for i in self.marked)
        return axes

    def __call__(self, ops: tuple) -> Tensor:
        if torch.compiler.is_compiling():
            return self.body(ops, *self.consts)
        axes = self._axes(len(ops))
        for i in axes:
            torch._dynamo.maybe_mark_dynamic(ops[i], 0)
        graph = _graph(self.body, self.mode, fixed_shapes(ops, axes))
        return graph(ops, *self.consts)

    def at(self, ops: tuple) -> Callable[[tuple], Tensor]:
        """The candidate for operands shaped as `ops` (token axes aside): the
        `CompiledOp` runner of their call pattern, resolved once per intra-op thread
        count and autocast state (the bucket fixes the rest of the pattern)."""
        if torch.compiler.is_compiling():
            return self
        body, consts = self.body, self.consts
        axes = self._axes(len(ops))
        graph = _graph(body, self.mode, fixed_shapes(ops, axes))
        kind = _device(ops).type
        runners: dict[tuple, Callable] = {}

        def run(ops: tuple) -> Tensor:
            if torch.compiler.is_compiling():
                return body(ops, *consts)
            for i in axes:
                torch._dynamo.maybe_mark_dynamic(ops[i], 0)
            ambient = _ambient(kind)
            runner = runners.get(ambient)
            if runner is None:
                runner = runners[ambient] = graph.resolved((ops, *consts))
            return runner(ops, *consts)

        return run


def _ambient(kind: str) -> tuple:
    """The call-pattern parts an operand bucket does not fix: intra-op threads and the
    autocast dtype on device type `kind`."""
    cast = torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else None
    return torch.get_num_threads(), cast


def compiled(op: Op, dynamic: str) -> CompiledBody:
    """The compile candidate of `op` (see `CompiledBody`)."""
    return CompiledBody(op, dynamic)


def _at(run: Callable, ops: tuple) -> Callable[[tuple], Tensor]:
    """`run` specialised to operands shaped as `ops` (a compile candidate), else as is."""
    return run.at(ops) if isinstance(run, CompiledBody) else run


def body_candidate(op: Op, dynamic: str) -> tuple[str, Callable[[tuple], Tensor]]:
    """The body candidate of `op`: ("compile", the compiled body), or ("eager", the eager
    body) for a body that runs through a custom autograd.Function (its compiled backward
    falls below eager's precision; see docs/internals/adapter-binding.md#body-candidate).
    """
    if op.body in bodies.AUTOGRAD_FUNCTION_BODIES:
        return "eager", eager(op)
    return "compile", compiled(op, dynamic)


def fixed(op: Op, run: Callable[[tuple], Tensor]) -> Callable:
    """Callable(t) -> `run` on the operands of runtime input `t`."""
    operands = op.operands

    def call(t):
        return run(operands(t))

    return call


def _layout(a) -> str:
    if a is None:
        return "none"
    grad = "g" if a.requires_grad else ""
    return f"{'x'.join(map(str, a.shape))}{short(a.dtype)}{grad}"


def problem_key(ops: tuple, tokens: tuple[int, ...]) -> str:
    """Tuning-table key of one call: each operand's shape (token axes bucketed), dtype
    and grad flag, and whether grad is enabled."""
    marked = {i % len(ops) for i in tokens}
    parts = []
    for i, a in enumerate(ops):
        if i in marked:
            n = pow2(a.shape[0], TOKEN_FLOOR)
            grad = "g" if a.requires_grad else ""
            rest = "x".join(map(str, a.shape[1:]))
            parts.append(f"{n}x{rest}{short(a.dtype)}{grad}")
        else:
            parts.append(_layout(a))
    return "|".join(parts) + f"-grad{int(torch.is_grad_enabled())}"


def _leaf(a, grad: bool):
    if not isinstance(a, Tensor):
        return a
    return a.detach().requires_grad_(grad and a.requires_grad)


def _launch(candidates: dict, ops: tuple, grad: bool, config: dict) -> Callable:
    """One timed call of a candidate on detached copies of `ops`: forward, plus backward
    over the operands that require grad when grad is enabled for the call."""
    fn = candidates[config["impl"]]
    leaves = tuple(_leaf(a, grad) for a in ops)
    tracked = [a for a in leaves if isinstance(a, Tensor) and a.requires_grad]
    if not tracked:
        return partial(fn, leaves)

    def run():
        out = fn(leaves)
        out.backward(torch.ones_like(out))
        for leaf in tracked:
            leaf.grad = None

    return run


def _device(ops: tuple) -> torch.device:
    return next(a.device for a in ops if isinstance(a, Tensor))


def _recomputed() -> bool:
    """Whether a call runs under saved-tensor hooks (a checkpointed forward or its
    recompute): what it binds there, its recompute must bind too."""
    return saved_tensor_hooks_active()


class AutotunedOp:
    """Callable(t) -> `op` on runtime input `t`, through the candidate the tuner settled
    for t's bucket. Where the tuner may not time (inference mode, a backward, a trace)
    an unsettled bucket runs the kernel and is asked again on its next call; in a
    checkpointed forward it binds the kernel for the process, so its recompute runs the
    same candidate whatever settles in between; under TUNE="off" every bucket settles
    on its stored winner, else the kernel."""

    def __init__(self, op: Op, backend: str, dynamic: str):
        self.op = op
        self.backend = backend
        self.name = f"lycoris_{op.name}_{backend}"
        impl, body = body_candidate(op, dynamic)
        self.body = body
        self.candidates = {"kernel": partial(op.kernel, backend), impl: body}
        # the kernel is the default; the body (`BODY`) is timed first
        self.configs = ({"impl": "kernel"}, {"impl": impl})
        self.bound: dict[tuple, Callable] = {}
        self._bucket = op.runtime.bucket
        self._operands = op.operands
        _OPS.add(self)

    def __call__(self, t):
        run = self.bound.get(self._bucket(t))
        if run is None:
            run = self.resolve(t)
        return run(self._operands(t))

    def resolve(self, t, cap: int | None = None) -> Callable[[tuple], Tensor]:
        """The candidate for t's bucket: the settled one (stored or measured now, each
        candidate's timing within `cap` bytes, else `OverBound`), kept once settled;
        the body candidate for operands no fused kernel takes (mixed fp16 / bf16, bf16
        below sm_80, off one CUDA device)."""
        if torch.compiler.is_compiling():
            return self.body
        ops = self._operands(t)
        if not fusable(ops):
            run = self.bound[self._bucket(t)] = _at(self.body, ops)
            return run
        device = _device(ops)
        key = problem_key(ops, self.op.tokens)
        grad = torch.is_grad_enabled()
        cfg = select(
            self.name,
            device,
            key,
            self._configs,
            partial(_launch, self.candidates, ops, grad),
            shortlist=len(self.configs),
            errors=ERRORS,
            autograd=True,
            lean=BODY,
            cap=cap,
        )
        run = _at(self.candidates[cfg["impl"]], ops)
        if decided(self.name, device, key) or _recomputed():
            self.bound[self._bucket(t)] = run
        return run

    def _configs(self) -> tuple:
        return self.configs

    def signature(self, dtype: torch.dtype, device) -> tuple:
        """What decides this op's problems apart from the token count: its name and the
        layout of every operand without a token axis."""
        with rng_kept([device]):
            t, _ = next(iter(self.op.runtime.examples([1], dtype, device)))
        ops = self._operands(t)
        marked = {i % len(ops) for i in self.op.tokens}
        kept = (_layout(a) for i, a in enumerate(ops) if i not in marked)
        return (self.name, *kept)

    def warm(
        self, rows: list[int], dtype: torch.dtype, device, budget: int | None = None
    ) -> int:
        """Settle every bucket of `rows` (or the fixed operand) on scratch inputs, one
        problem at a time: its inputs are built when it is reached and dropped after
        it, and a measured problem's free blocks go back to the driver. With `budget`
        (bytes a problem's timing may add), a token bucket whose working memory,
        scaled from the largest measured bucket of its grad pattern by the rows, passes
        it, or whose timing does, is never timed: it inherits that bucket's decision
        (`tuner.inherit`); the smallest bucket of a pattern is always timed. Returns
        the problems measured (a stored or settled problem is only bound). The scratch
        inputs never move the caller's generators (`rng_kept`)."""
        with rng_kept([device]):
            return self._warm(rows, dtype, device, budget)

    def _warm(self, rows: list[int], dtype: torch.dtype, device, budget) -> int:
        measured, fits = 0, {}
        for build, grad in self.op.runtime.problems(rows):
            with torch.set_grad_enabled(grad):
                outcome = self._settle(build(dtype, device), budget, fits)
            if outcome == "measured":
                measured += 1
                release(device)
        return measured

    def _pattern(self, t, ops: tuple) -> tuple[tuple | None, int]:
        """(t's bucket without its token count, its token rows); (None, 0) for an op
        without a token axis."""
        if not self.op.tokens:
            return None, 0
        return self._bucket(t)[1:], ops[self.op.tokens[0] % len(ops)].shape[0]

    def _settle(self, t, budget: int | None, fits: dict) -> str:
        """Bind t's bucket: "known" (settled or stored already), "inherited" (over the
        budget, from `fits`: pattern -> (rows, key) of its largest measured bucket) or
        "measured"."""
        ops = self._operands(t)
        if not fusable(ops):
            self.resolve(t)
            return "known"
        at, key = _device(ops), problem_key(ops, self.op.tokens)
        pattern, rows = self._pattern(t, ops)
        known = decided(self.name, at, key) or stored(self.name, at, key)
        source = fits.get(pattern) if budget is not None else None
        if not known and source is not None and self._over(at, source, rows, budget):
            return self._inherit(t, at, key, source)
        try:
            self.resolve(t, None if known else budget)
        except OverBound:
            release(at)
            if source is not None:
                return self._inherit(t, at, key, source)
            self.resolve(t)
        stored_row = row(self.name, at, key) or {}
        if pattern is not None and "inherited" not in stored_row:
            fits[pattern] = (rows, key)
        return "known" if known else "measured"

    def _over(self, device, source: tuple[int, str], rows: int, budget: int) -> bool:
        """Whether `rows` rows would pass `budget`: the source bucket's largest working
        memory scaled by the rows."""
        src_rows, src_key = source
        used = (row(self.name, device, src_key) or {}).get("working_mib") or {}
        mib = max(used.values(), default=0.0)
        return mib * MIB * rows / max(src_rows, 1) > budget

    def _inherit(self, t, device, key: str, source: tuple[int, str]) -> str:
        inherit(self.name, device, key, source[1])
        self.resolve(t)
        return "inherited"


def reset() -> None:
    """Forget every bound candidate (tuner reset, broadcast, tests)."""
    for op in list(_OPS):
        op.bound.clear()


on_reset(reset)
