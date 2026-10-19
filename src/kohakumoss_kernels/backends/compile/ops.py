"""torch.compile of the eager reference ops, one `CompiledOp` per op for the process."""

import types
from collections.abc import Callable
from functools import cache, partial

import torch
import torch._dynamo
import torch.nn.functional as F
from torch import Tensor

from kohakumoss_common.log import get_logger
from kohakumoss_kernels import compile_setup
from kohakumoss_kernels.backends.eager import ffn, hadamard, loss, norm, rope, textenc
from kohakumoss_kernels.errors import unavailable

log = get_logger(__name__)


def _linear_situ(y: Tensor, w: Tensor, bias: Tensor | None) -> Tensor:
    return ffn.situ_glu(
        F.linear(y, w.to(y.dtype), None if bias is None else bias.to(y.dtype))
    )


def _linear_gated_residual(a, w, bias, x, gate, seg, r):
    o = F.linear(a, w.to(a.dtype), None if bias is None else bias.to(a.dtype))
    return norm.gated_residual(x, o, gate, seg, r)


def _linear_qk_rope(y, wq, wk, q_weight, k_weight, pos, bands, eps, heads):
    q = F.linear(y, wq.to(y.dtype)).unflatten(-1, (heads, -1))
    k = F.linear(y, wk.to(y.dtype)).unflatten(-1, (heads, -1))
    return rope.qk_norm_rope(q, k, q_weight, k_weight, pos, bands, eps)


EAGER: dict[str, Callable] = {
    "rms_norm": norm.rms_norm,
    "rms_norm_mod": norm.rms_norm_mod,
    "gated_residual": norm.gated_residual,
    "qk_norm_rope": rope.qk_norm_rope,
    "situ_glu": ffn.situ_glu,
    "token_mse": loss.token_mse,
    "linear_situ": _linear_situ,
    "linear_gated_residual": _linear_gated_residual,
    "linear_qk_rope": _linear_qk_rope,
    "causal_conv_varlen": textenc.causal_conv_varlen,
    "rms_norm_gated": textenc.rms_norm_gated,
    "group_hadamard": hadamard.group_hadamard,
}
COMPILED = tuple(EAGER)


GRAPH_LIMIT = 8
RECOMPILE_LIMIT = 64
ACCUMULATED_LIMIT = 4096


class GraphLimitError(RuntimeError):
    """A compiled op needed more graphs for one call pattern than GRAPH_LIMIT."""


def _raise_dynamo_limits() -> None:
    """Raise Dynamo's recompile limits above GRAPH_LIMIT."""
    cfg = torch._dynamo.config
    cfg.recompile_limit = max(cfg.recompile_limit, RECOMPILE_LIMIT)
    cfg.accumulated_recompile_limit = max(
        cfg.accumulated_recompile_limit, ACCUMULATED_LIMIT
    )


def _flat(args) -> list:
    """The leaves of `args`, tuple and list arguments expanded in order."""
    out = []
    for a in args:
        if isinstance(a, (tuple, list)):
            out.extend(_flat(a))
        else:
            out.append(a)
    return out


def _pattern(args) -> tuple:
    """What Dynamo guards on besides shapes: the device, grad mode, intra-op threads,
    autocast on the arguments' device and, per argument, tensor dtype / requires_grad /
    rank, None and Python scalars (inside tuple / list arguments too). See
    docs/internals/kernel-dev.md#compile-backend."""
    args = _flat(args)
    tensors = [a for a in args if isinstance(a, Tensor)]
    device = tensors[0].device if tensors else torch.device("cpu")
    autocast = torch.is_autocast_enabled(device.type)
    keys = [
        device,
        torch.is_grad_enabled(),
        torch.get_num_threads(),
        torch.get_autocast_dtype(device.type) if autocast else None,
    ]
    for a in args:
        if isinstance(a, Tensor):
            keys.append((a.dtype, a.requires_grad, a.dim()))
        elif a is None or isinstance(a, (bool, int, float, str)):
            keys.append(a)
        else:
            keys.append(type(a).__name__)
    return tuple(keys)


def _fresh(fn: Callable, tag: str) -> Callable:
    """A copy of `fn` with its own code object (named `<name>_<tag>`), hence its own
    Dynamo cache."""
    code = fn.__code__.replace(co_name=f"{fn.__code__.co_name}_{tag}")
    copy = types.FunctionType(
        code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__
    )
    copy.__kwdefaults__ = fn.__kwdefaults__
    return copy


def _counting(backend: str | Callable, name: str, key: tuple) -> Callable:
    """`backend` that counts its graphs and raises past GRAPH_LIMIT."""
    inner = (
        torch._dynamo.lookup_backend(backend) if isinstance(backend, str) else backend
    )
    graphs = [0]

    def compile_graph(gm, example_inputs):
        graphs[0] += 1
        if graphs[0] > GRAPH_LIMIT:
            raise GraphLimitError(
                f"{name}: more than {GRAPH_LIMIT} graphs for call pattern {key}"
            )
        return inner(gm, example_inputs)

    return compile_graph


class CompiledOp:
    """torch.compile of one eager reference op (`EAGER[name]`, or the function `eager`):
    one compiled callable (own code object and Dynamo cache) per call pattern
    (`_pattern`); a pattern needing more than GRAPH_LIMIT graphs runs the eager
    reference for the rest of the process (`_limit`, one WARNING, never raised), a
    build failure of the first call raises `BackendUnavailable`. Inside an
    outer torch.compile the eager reference is traced. `resolved(args)` is the runner of
    `args`' pattern, for callers that bind per pattern themselves.
    See docs/internals/kernel-dev.md#compile-backend."""

    def __init__(
        self,
        name: str,
        dynamic: bool | None,
        backend: str | Callable,
        eager: Callable | None = None,
    ):
        self.eager = EAGER[name] if eager is None else eager
        self.dynamic = dynamic
        self.backend = backend
        self.__name__ = f"compiled_{name}"
        self.compiled: dict[tuple, Callable] = {}
        self.functions: dict[tuple, Callable] = {}
        self.runners: dict[tuple, Callable] = {}
        self.limited: set[tuple] = set()
        self.built = False

    def __call__(self, *args):
        if torch.compiler.is_compiling():
            return self.eager(*args)
        return self.resolved(args)(*args)

    def resolved(self, args) -> Callable:
        """The runner of `args`' call pattern (compiled on first use), with the build
        failure translation of `__call__`."""
        key = _pattern(args)
        run = self.runners.get(key)
        if run is None:
            run = self.runners[key] = partial(self._run, key, self._compile(key))
        return run

    def _run(self, key: tuple, fn: Callable, *args):
        if torch.compiler.is_compiling() or key in self.limited:
            return self.eager(*args)
        try:
            out = fn(*args)
        except torch._dynamo.exc.BackendCompilerFailed as err:
            if isinstance(err.inner_exception, GraphLimitError):
                return self._limit(key, err.inner_exception, args)
            if self.built:
                raise
            raise unavailable(self.__name__, err) from err
        except Exception as err:  # noqa: BLE001
            if self.built:
                raise
            raise unavailable(self.__name__, err) from err
        self.built = True
        return out

    def _limit(self, key: tuple, err: GraphLimitError, args):
        """Call pattern `key` passed GRAPH_LIMIT graphs: it runs the eager reference
        for the rest of the process (one WARNING), this call included."""
        self.limited.add(key)
        log.warning(f"{err}; this call pattern runs the eager reference from now on")
        return self.eager(*args)

    def _compile(self, key: tuple) -> Callable:
        """The compiled callable of call pattern `key` (a fresh code object), under
        this package's compile settings (`compile_setup`)."""
        compile_setup.apply()
        _raise_dynamo_limits()
        fresh = _fresh(self.eager, f"pattern{len(self.compiled)}")
        fn = torch.compile(
            fresh,
            dynamic=self.dynamic,
            backend=_counting(self.backend, self.__name__, key),
        )
        self.functions[key] = fresh
        self.compiled[key] = fn
        return fn


@cache
def compiled_op(
    name: str, dynamic: bool | None = True, backend: str | Callable = "inductor"
) -> CompiledOp:
    """The process's `CompiledOp` of `name` for (`dynamic`, `backend`)."""
    return CompiledOp(name, dynamic, backend)
