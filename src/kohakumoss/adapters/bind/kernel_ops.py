"""The kernel ops one bind of a layer draws from. `FixedBackendOps(backend)` runs every
op on that backend (its fused kernel, the compiled body or the eager body);
`AutotunedOps(backend)` binds each op to an `AutotunedOp`, the fused kernel of `backend`
against the compiled body. Both expose the same factories; each returns the callable a
forward wrapper takes, or None where the layer has no such op. See
docs/internals/adapter-binding.md#kernel-ops."""

from collections.abc import Callable

from lycoris.kernels.dispatch import FUSED
from torch import Tensor

from kohakumoss.adapters.bind import forwards, ops, weight_ops
from kohakumoss.adapters.bind.autotuned import (
    AutotunedOp,
    Op,
    body_candidate,
    eager,
    fixed,
)


def on_tokens(run: Callable | None) -> Callable | None:
    """`run` on a (tokens, channels) view of its activation, reshaped back."""
    if run is None:
        return None

    def call(x: Tensor) -> Tensor:
        y = run(x.reshape(-1, x.shape[-1]))
        return y.reshape(*x.shape[:-1], y.shape[-1])

    return call


class KernelOps:
    """The factories shared by both variants; `bind(op)` is the variant's choice.
    `fuse_base_add`: the base add goes inside the op (the *_with_base / merged
    factories, None without it); `dynamic`: the compile candidate's shape mode."""

    def __init__(
        self, backend: str, fuse_base_add: bool = True, dynamic: str = "tokens"
    ):
        self.backend = backend
        self.fuse_base_add = fuse_base_add
        self.dynamic = dynamic
        self.autotuned: list[AutotunedOp] = []

    def bind(self, op: Op) -> Callable | None:
        raise NotImplementedError

    def _with_base(self, build, *args) -> Callable | None:
        return self.bind(build(*args)) if self.fuse_base_add else None

    def lora(self, lyco, gamma):
        return on_tokens(self.bind(ops.lora_bypass(lyco, gamma)))

    def loha(self, lyco, gamma):
        return on_tokens(self.bind(ops.loha_bypass(lyco, gamma)))

    def lokr(self, lyco, scale):
        return on_tokens(self.bind(ops.lokr_bypass(lyco, scale)))

    def lora_with_base(self, lyco, gamma):
        return self._with_base(ops.lora_bypass_with_base, lyco, gamma)

    def loha_with_base(self, lyco, gamma):
        return self._with_base(ops.loha_bypass_with_base, lyco, gamma)

    def lokr_with_base(self, lyco, scale):
        return self._with_base(ops.lokr_bypass_with_base, lyco, scale)

    def oft(self, layer, lyco):
        """The whole OFT forward: the rotated base output (`layer` a weak reference
        to the layer)."""
        if self.fuse_base_add:
            rotate = on_tokens(self.bind(ops.oft_rotate(lyco)))
            return None if rotate is None else forwards.rotated(layer, rotate)
        delta = on_tokens(self.bind(ops.oft_bypass(lyco)))
        return None if delta is None else forwards.plus_rotation(layer, delta)

    def boft(self, layer, lyco, m):
        """The whole BOFT forward at multiplier `m`: the rotated base output."""
        if self.fuse_base_add:
            rotate = on_tokens(self.bind(ops.boft_rotate(lyco, m)))
            return None if rotate is None else forwards.rotated(layer, rotate)
        delta = on_tokens(self.bind(ops.boft_bypass(lyco, m)))
        return None if delta is None else forwards.plus_rotation(layer, delta)

    def ia3_scale(self, lyco, m):
        return on_tokens(self.bind(ops.ia3_bypass_scale(lyco, m)))

    def lora_weight(self, lyco):
        return self.bind(weight_ops.lora_weight(lyco))

    def loha_weight(self, lyco):
        return self.bind(weight_ops.loha_weight(lyco))

    def lokr_weight(self, lyco):
        return self.bind(weight_ops.lokr_weight(lyco))

    def oft_weight(self, lyco):
        return self.bind(weight_ops.oft_weight(lyco))

    def boft_weight(self, lyco, m):
        return self.bind(weight_ops.boft_weight(lyco, m))

    def ia3_weight(self, lyco, m):
        return self.bind(weight_ops.ia3_weight(lyco, m))

    def glora_weight(self, lyco):
        return self.bind(weight_ops.glora_weight(lyco))

    def lora_merged_weight(self, lyco):
        return self._with_base(weight_ops.lora_merged_weight, lyco)

    def loha_merged_weight(self, lyco):
        return self._with_base(weight_ops.loha_merged_weight, lyco)

    def lokr_merged_weight(self, lyco):
        return self._with_base(weight_ops.lokr_merged_weight, lyco)

    def oft_merged_weight(self, lyco):
        return self._with_base(weight_ops.oft_merged_weight, lyco)

    def decompose(self, lyco, on_out):
        return self.bind(weight_ops.decompose(lyco, on_out))

    def decompose_with_delta(self, lyco, on_out):
        return self._with_base(weight_ops.decompose_with_delta, lyco, on_out)


class FixedBackendOps(KernelOps):
    """Every op on `backend`: a fused backend's kernel (None outside its scope), the
    compiled body (`compile`) or the eager body (`torch`)."""

    def bind(self, op: Op) -> Callable | None:
        match self.backend:
            case "torch":
                return fixed(op, eager(op))
            case "compile":
                return fixed(op, body_candidate(op, self.dynamic)[1])
        if not op.in_scope(self.backend):
            return None
        kernel, backend = op.kernel, self.backend

        def run(o: tuple) -> Tensor:
            return kernel(backend, o)

        return fixed(op, run)


class AutotunedOps(KernelOps):
    """Every op an `AutotunedOp` over `backend`'s fused kernel and the body candidate;
    an op outside the kernel's scope on this layer runs the body candidate."""

    def bind(self, op: Op) -> Callable | None:
        if not op.in_scope(self.backend):
            return fixed(op, body_candidate(op, self.dynamic)[1])
        tuned = AutotunedOp(op, self.backend, self.dynamic)
        self.autotuned.append(tuned)
        return tuned


def kernel_ops(
    backend: str, autotune: bool, fuse_base_add: bool = True, dynamic: str = "tokens"
) -> KernelOps:
    """The kernel ops of one bind: autotuned when the configured backend is auto and
    `backend` has fused kernels, else fixed to `backend`."""
    if autotune and backend in FUSED:
        return AutotunedOps(backend, fuse_base_add, dynamic)
    return FixedBackendOps(backend, fuse_base_add, dynamic)
