"""Per-shape choice of `attention_varlen`: a backend's varlen kernel against torch's
flash `varlen_attn` (where the build and card have it; else the per-segment SDPA loop,
the eager reference, never used inside torch.compile). The tuner times, once per
problem key, forward + backward on leaf copies where a backward will run (`-g1` keys),
the forward alone where none will (`-g0`), each call on a shape of its own, and binds
the winner; later calls only look it up. See docs/internals/tuning.md.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from kohakumoss_kernels import tuning
from kohakumoss_kernels.backends.eager.attention import attention_varlen as sdpa_loop
from kohakumoss_kernels.errors import BUILD_ERRORS, unavailable
from kohakumoss_kernels.saved_ops import keep_named
from kohakumoss_kernels.tuning.buckets import mean_bucket
from kohakumoss_kernels.tuning.tuner import on_reset

try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn
except ImportError:
    _varlen_attn = None

if _varlen_attn is not None:
    keep_named("torch_attn", "_varlen_attn")
    keep_named("aten", "_flash_attention_forward")
JITTER = 64  # shape variants a timed problem cycles through (see `_variants`)


def flash_varlen_usable(device: torch.device) -> bool:
    """Whether torch's flash `varlen_attn` runs on `device`: the build has flash attention
    and the card is sm80+ (the Linux torch builds carry it)."""
    return (
        _varlen_attn is not None
        and device.type == "cuda"
        and torch.backends.cuda.is_flash_attention_available()
        and torch.cuda.get_device_capability(device) >= (8, 0)
    )


def torch_varlen(q, k, v, cu, max_len, scale):
    """torch's flash varlen attention with one `cu_seqlens` for queries and keys."""
    return _varlen_attn(q, k, v, cu, cu, max_len, max_len, scale=scale)


def _variants(q: Tensor, cu: Tensor) -> list[tuple[int, Tensor]]:
    """(tokens, cu_seqlens) of the problem with its last segment cut by 0, 1, 2, ...
    tokens (at most JITTER variants, the segment kept non-empty): every timed call runs
    a shape of its own, as training does, so a backend that plans per shape (cuDNN)
    pays its host planning cost in the measurement."""
    bounds = cu.tolist()
    room = max(min(JITTER, bounds[-1] - bounds[-2]), 1)
    out = []
    for cut in range(room):
        shorter = cu.clone()
        shorter[-1] -= cut
        out.append((q.shape[0] - cut, shorter))
    return out


def _launch(fn, q, k, v, cu, max_len, scale, backward: bool) -> Callable[[], None]:
    """A tuner launch of `fn` on copies of q, k, v, cycling through the problem's
    shape variants (`_variants`): forward + backward on leaves with `backward`, else the
    forward under no_grad."""
    variants = _variants(q, cu)
    turn = [0]
    if not backward:
        copies = [t.detach() for t in (q, k, v)]

        def forward():
            n, c = variants[turn[0] % len(variants)]
            turn[0] += 1
            with torch.no_grad():
                fn(*(t[:n] for t in copies), c, max_len, scale)

        return forward
    base = [t.detach().clone() for t in (q, k, v)]
    grad = torch.randn_like(q)
    leaves = [
        ([t[:n].detach().requires_grad_() for t in base], grad[:n], c)
        for n, c in variants
    ]

    def run():
        inputs, g, c = leaves[turn[0] % len(leaves)]
        turn[0] += 1
        fn(*inputs, c, max_len, scale).backward(g)
        for t in inputs:
            t.grad = None

    return run


def needs_backward(*ts: Tensor) -> bool:
    """A backward will run through this call (grad mode on, some input requires grad)."""
    return torch.is_grad_enabled() and any(t.requires_grad for t in ts)


class VarlenChoice:
    """`attention_varlen` through the fastest of `kernel` and the PyTorch paths per
    problem key `key(q, segments, max_seqlen)` plus the mean-segment-length bucket
    (`-M<tokens>`) and whether a backward runs (`-g0` / `-g1`), timed by wall time
    over varying shapes (host launches and per-shape planning included, which a device
    timer does not see); the tuning table is `attn_varlen_shapes_<backend>`. Inside
    torch.compile it runs the kernel (no per-shape key, so no shape specialisation)."""

    def __init__(self, backend: str, kernel: Callable, key: Callable):
        self.backend = backend
        self.kernel = kernel
        self.key = key
        self.table = f"attn_varlen_shapes_{backend}"
        self.bound: dict[tuple, Callable] = {}
        self.__name__ = "attention_varlen"
        on_reset(self.bound.clear)

    def problem(self, q: Tensor, segments: int, max_seqlen: int, backward: bool) -> str:
        """The choice's table key of one call."""
        mean = mean_bucket(q.shape[0], segments)
        return f"{self.key(q, segments, max_seqlen)}-M{mean}-g{int(backward)}"

    def impls(self, device: torch.device) -> dict[str, Callable]:
        """The candidates on `device`: the kernel and torch varlen; the SDPA loop (it
        reads the segment bounds on the host: a stream sync per call) only where torch
        varlen does not run."""
        impls = {self.backend: self.kernel}
        if flash_varlen_usable(device):
            impls["torch_varlen"] = torch_varlen
        else:
            impls["sdpa"] = sdpa_loop
        return impls

    def _build(self, q, k, v, cu, max_len, scale, key, backward: bool) -> Callable:
        impls = self.impls(q.device)
        names = list(impls)
        choice = tuning.select(
            self.table,
            q.device,
            key,
            lambda: [{"impl": name} for name in names],
            lambda c: _launch(impls[c["impl"]], q, k, v, cu, max_len, scale, backward),
            shortlist=len(names),
            errors=BUILD_ERRORS,
            autograd=True,
            timing="wall",
        )
        return impls[choice["impl"]]

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        scale: float,
    ) -> Tensor:
        if torch.compiler.is_compiling():
            return self.kernel(q, k, v, cu_seqlens, max_seqlen, scale)

        max_seqlen, scale = int(max_seqlen), float(scale)
        backward = needs_backward(q, k, v)
        key = self.problem(q, cu_seqlens.shape[0] - 1, max_seqlen, backward)
        ident = (q.device, key)
        try:
            fn = self.bound.get(ident)
            if fn is None:
                args = (q, k, v, cu_seqlens, max_seqlen, scale, key, backward)
                fn = self._build(*args)
                if tuning.decided(self.table, q.device, key):
                    self.bound[ident] = fn
            return fn(q, k, v, cu_seqlens, max_seqlen, scale)
        except Exception as err:
            raise unavailable(
                "attention_varlen", err, f"{self.backend} attention choice"
            ) from err


def varlen_choice(backend: str, kernel_fn: Callable, key_fn: Callable) -> VarlenChoice:
    """The `attention_varlen` registry entry of `backend` over its kernel `kernel_fn`."""
    return VarlenChoice(backend, kernel_fn, key_fn)
