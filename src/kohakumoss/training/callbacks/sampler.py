"""Wall-time attribution by stack sampling: a daemon thread samples every thread's
Python stack at a fixed interval and charges each sample to the first matching
category of `CATEGORIES` (most specific first), else to `other` keyed by its innermost
frame; a frame's category index is matched once per (module, function) and looked up
after. Work in native code is charged to the Python frame that called it."""

import sys
import threading
from collections import Counter

INTERVAL = 0.005

# (category, module prefix, function names or None for any function of the module)
CATEGORIES: tuple[tuple[str, str, frozenset | None], ...] = (
    ("torch_compile", "torch._dynamo", None),
    ("torch_compile", "torch._inductor", None),
    ("kernel_tuning", "kohakumoss_kernels.tuning.timing", None),
    ("kernel_tuning", "kohakumoss_kernels.tuning.tuner", frozenset({"_measure"})),
    ("compile_probe", "kohakumoss_kernels.backends.compile.choice", frozenset({"_probe"})),
    ("fp8_probe", "kohakumoss_kernels.backends.triton.fp8.linear", frozenset(
        {"_probe", "_select"})),
    ("triton_jit_load", "triton.compiler", None),
    ("triton_jit_load", "triton.runtime.jit", frozenset({"_do_compile", "create_binder"})),
    ("triton_jit_load", "triton.runtime.build", None),
    ("tilelang_jit_load", "tilelang", frozenset({"compile", "lower", "_compile"})),
    ("tilelang_jit_load", "kohakumoss_kernels.backends.tilelang", frozenset(
        {"compiled", "_compiled", "_build"})),
    ("stream_setup", "kohakumoss.offload", frozenset({"__init__", "open", "_alloc"})),
    ("stream_wait", "kohakumoss.offload", frozenset(
        {"_wait_device", "_load", "_copy_device", "_copy_host", "_prefetch", "enter"})),
    ("text_encode", "kohakumoss.encoders.text", None),
    ("text_encode", "kohakumoss.offload.encoders.text", None),
    ("vae_encode", "kohakumoss.encoders.vae", None),
    ("vae_encode", "kohakumoss.offload.encoders.vae", None),
    ("data", "kohakumoss_data", None),
    ("optimizer", "torch.optim", None),
    ("optimizer", "kohakumoss.training.optim", None),
    ("forward_backward", "torch.autograd", None),
    ("forward_backward", "kohakumoss.training.step", None),
)  # fmt: skip


def rank_of(module: str, func: str) -> int | None:
    """Index of the first CATEGORIES entry a frame (module, function) matches."""
    for n, (_, prefix, funcs) in enumerate(CATEGORIES):
        if module.startswith(prefix) and (funcs is None or func in funcs):
            return n
    return None


def _best(ranks) -> int | None:
    found = [r for r in ranks if r is not None]
    return min(found) if found else None


def classify(stack: list[tuple[str, str]]) -> str:
    """The category of one sampled stack (innermost frame first): the first CATEGORIES
    entry any frame matches, else `other:` its innermost frame."""
    best = _best(rank_of(module, func) for module, func in stack)
    if best is not None:
        return CATEGORIES[best][0]
    module, func = stack[0] if stack else ("?", "?")
    return f"other:{module}.{func}"


def _module_of(frame) -> str:
    return frame.f_globals.get("__name__", "?")


MISS = object()


def _thread_name(ident: int) -> str:
    thread = threading._active.get(ident)
    return thread.name if thread is not None else str(ident)


class StackSampler:
    """Samples every thread of the process from `start()` to `stop()`; `report()` maps
    (thread name, category) to seconds (samples x `interval`). `frames()` gives
    {thread ident: innermost frame} and `name(ident)` a thread's name (the process's
    own by default)."""

    def __init__(
        self,
        interval: float = INTERVAL,
        frames=sys._current_frames,
        name=_thread_name,
    ):
        self.interval = interval
        self.frames = frames
        self.name = name
        self.counts: Counter = Counter()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.me = None
        self.names: dict[int, str] = {}
        self.ranks: dict = {}

    def start(self) -> "StackSampler":
        self.thread.start()
        return self

    def sample(self) -> None:
        """Charge one sample of every thread but the sampler's own."""
        for ident, frame in self.frames().items():
            if ident == self.me:
                continue
            if ident not in self.names:
                self.names[ident] = self.name(ident)
            self.counts[(self.names[ident], self.category(frame))] += 1

    def _rank(self, frame) -> int | None:
        """The frame's CATEGORIES index, matched once per (module, function)."""
        key = (_module_of(frame), frame.f_code.co_name)
        rank = self.ranks.get(key, MISS)
        if rank is MISS:
            rank = self.ranks[key] = rank_of(*key)
        return rank

    def category(self, frame) -> str:
        """`classify` of the stack ending in `frame`, one dict lookup per frame."""
        best, inner = None, frame
        while frame is not None and best != 0:
            rank = self._rank(frame)
            if rank is not None and (best is None or rank < best):
                best = rank
            frame = frame.f_back
        if best is not None:
            return CATEGORIES[best][0]
        return f"other:{_module_of(inner)}.{inner.f_code.co_name}"

    def _run(self) -> None:
        self.me = threading.get_ident()
        while not self.stopped.wait(self.interval):
            self.sample()

    def stop(self) -> None:
        self.stopped.set()
        self.thread.join(timeout=1.0)

    def report(self) -> dict[tuple[str, str], float]:
        return {key: n * self.interval for key, n in self.counts.items()}


def main_thread_name() -> str:
    return threading.main_thread().name
