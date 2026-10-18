"""Kernel-vs-PyTorch choice for the optimizer step ops: per problem key (tensor dtypes,
string options, bucketed element count) the backend kernel and the reference step are
timed once on scratch copies, the winner stored per card and bound."""

from collections.abc import Callable

from torch import Tensor

from kohakumoss_kernels import tuning
from kohakumoss_kernels.tuning.buckets import pow2
from kohakumoss_kernels.tuning.tuner import on_reset

CHOICES = ({"impl": "kernel"}, {"impl": "torch"})
NUMEL_FLOOR = 4096


def _first(a) -> Tensor | None:
    match a:
        case Tensor():
            return a
        case list() | tuple():
            return next((t for t in a if isinstance(t, Tensor)), None)
    return None


def _numel(a) -> int:
    match a:
        case Tensor():
            return a.numel()
        case list() | tuple():
            return sum(t.numel() for t in a if isinstance(t, Tensor))
    return 0


def _scratch(a):
    match a:
        case Tensor():
            return a.detach().clone()
        case list():
            return [_scratch(t) for t in a]
        case tuple():
            return tuple(_scratch(t) for t in a)
    return a


def problem_key(args: tuple, kwargs: dict) -> str:
    """Dtypes of the tensor arguments, string options and the bucketed sizes."""
    parts = []
    for a in (*args, *kwargs.values()):
        t = _first(a)
        if t is not None:
            parts.append(str(t.dtype).removeprefix("torch."))
        elif isinstance(a, str):
            parts.append(a)
    head = args[0]
    count = len(head) if isinstance(head, (list, tuple)) else 1
    return f"{'-'.join(parts)}-t{pow2(count, 1)}-n{pow2(_numel(head), NUMEL_FLOOR)}"


class KernelOrReference:
    """`kernel` or `reference` per problem key, decided by one timed measurement."""

    def __init__(self, backend: str, name: str, kernel: Callable, reference: Callable):
        self.name = f"choice_{backend}_{name}"
        self.impls = {"kernel": kernel, "torch": reference}
        self.bound: dict[tuple, Callable] = {}
        self.__wrapped__ = kernel
        self.__name__ = getattr(kernel, "__name__", name)
        on_reset(self.bound.clear)

    def _decide(self, args: tuple, kwargs: dict, key: str) -> Callable:
        device = _first(args[0]).device
        scratch: list[tuple[list, dict]] = []

        def launch(cfg: dict) -> Callable[[], None]:
            # scratch copies are made only when a config is timed
            if not scratch:
                copies = {k: _scratch(v) for k, v in kwargs.items()}
                scratch.append(([_scratch(a) for a in args], copies))
            fn, (a, kw) = self.impls[cfg["impl"]], scratch[0]
            return lambda: fn(*a, **kw)

        cfg = tuning.select(
            self.name,
            device,
            key,
            lambda: list(CHOICES),
            launch,
            shortlist=len(CHOICES),
            in_backward=True,
        )
        fn = self.impls[cfg["impl"]]
        if tuning.decided(self.name, device, key):
            self.bound[(device, key)] = fn
        return fn

    def __call__(self, *args, **kwargs) -> None:
        first = _first(args[0])
        if first is None:
            return
        key = problem_key(args, kwargs)
        fn = self.bound.get((first.device, key))
        if fn is None:
            fn = self._decide(args, kwargs, key)
        fn(*args, **kwargs)


def kernel_or_reference(
    backend: str, name: str, kernel: Callable, reference: Callable
) -> KernelOrReference:
    """The registry entry of an optimizer step kernel: chosen against `reference`."""
    return KernelOrReference(backend, name, kernel, reference)
