"""Kernel-vs-compile binding on CPU: a bound compile path calls its pattern's runner (no
pattern built per call), a call under torch.compile tracing binds nothing and builds no
per-call key (pinned or not), the compile
call pattern carries the device, and a retune whose inputs do not fit in memory keeps
the kernel instead of raising."""

import pytest
import torch

from kohakumoss_kernels.backends.compile import choice, ops
from kohakumoss_kernels.backends.eager import ffn
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    tuner._TABLES.clear()
    tuner.reset()
    yield
    tuner._TABLES.clear()
    tuner.reset()


def _op() -> choice.KernelOrCompile:
    op = choice.kernel_or_compile("test", "situ_glu", ffn.situ_glu, choice.shape_key)
    op.impls["compile"] = ops.CompiledOp("situ_glu", True, "eager")
    return op


def _counting(monkeypatch) -> list:
    calls = []
    real = ops._pattern

    def pattern(args):
        calls.append(1)
        return real(args)

    monkeypatch.setattr(ops, "_pattern", pattern)
    return calls


def check_no_pattern_per_call(op, calls) -> None:
    x = torch.randn(64, 32)
    op(x)
    first = len(calls)
    for _ in range(3):
        op(x)
    assert len(calls) == first, f"{len(calls) - first} patterns built in bound calls"


@pytest.mark.budget(60)
def test_bound_compile_skips_the_pattern(fresh, monkeypatch):
    op = _op()
    op.bind("compile")
    check_no_pattern_per_call(op, _counting(monkeypatch))


@pytest.mark.budget(60)
def test_negative_generic_compiled_op(fresh, monkeypatch):
    op = _op()
    op.bind("compile")
    monkeypatch.setattr(
        choice.KernelOrCompile, "_impl", lambda self, i, a: self.impls[i]
    )
    must_fail(
        lambda: check_no_pattern_per_call(op, _counting(monkeypatch)),
        "pattern per call",
    )


def _untraceable(args):
    raise AssertionError(
        "a traced call built the per-call key (torch.get_num_threads breaks a "
        "fullgraph trace)"
    )


def check_traced_binds_nothing(op, monkeypatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(choice.torch.compiler, "is_compiling", lambda: True)
        patch.setattr(choice, "_call_key", _untraceable)
        op(torch.randn(16, 8))
    assert not op.bound, "a traced call bound the kernel for eager calls"


@pytest.mark.parametrize("pinned", (None, "kernel", "compile"))
def test_traced_call_binds_nothing(fresh, monkeypatch, pinned):
    op = _op()
    if pinned is not None:
        op.bind(pinned)
    check_traced_binds_nothing(op, monkeypatch)


class _StoringWhenTraced(choice.KernelOrCompile):
    def __call__(self, *args):
        fn = self.impls["kernel"]
        self.bound[("traced",)] = fn
        return fn(*args)


class _KeyedWhenTraced(choice.KernelOrCompile):
    def __call__(self, *args):
        qk = choice._call_key(args)
        return (self.bound.get(qk) or self._lookup(args, qk))(*args)


@pytest.mark.parametrize("broken", (_StoringWhenTraced, _KeyedWhenTraced))
def test_negative_traced_call_binds(fresh, monkeypatch, broken):
    op = broken("test", "situ_glu", ffn.situ_glu, choice.shape_key)
    must_fail(lambda: check_traced_binds_nothing(op, monkeypatch), broken.__name__)


def check_pattern_has_device(pattern) -> None:
    """The pattern holds the arguments' device (two GPUs get two compiled callables)."""
    t = torch.empty(4, 8)
    assert t.device in pattern((t,)), "the call pattern does not name the device"


def test_pattern_carries_the_device():
    check_pattern_has_device(ops._pattern)


def test_negative_pattern_without_device():
    must_fail(lambda: check_pattern_has_device(lambda a: ops._pattern(a)[1:]), "device")


def _oom(layout):
    raise torch.OutOfMemoryError("CUDA out of memory (test)")


def check_retune_oom_kept(op, monkeypatch) -> None:
    tuner.set_mode("table")
    x = torch.randn(64, 32)
    op(x)
    assert op.pending, "first sight of a problem is queued"
    monkeypatch.setattr(choice, "_rebuild", _oom)
    try:
        done = choice.retune_pending()
    except torch.OutOfMemoryError as err:
        raise AssertionError(f"a retune OOM reached the step: {err}") from err
    assert done == 0 and not op.pending
    assert op(x) is not None and op.choice is None


def test_retune_oom_keeps_the_kernel(fresh, monkeypatch):
    check_retune_oom_kept(_op(), monkeypatch)


def test_negative_retune_oom_raises(fresh, monkeypatch):
    def unguarded(self, layouts):
        self.decide(*[choice._rebuild(s) for s in layouts])
        return "decided"

    monkeypatch.setattr(choice.KernelOrCompile, "_retune_one", unguarded)
    must_fail(lambda: check_retune_oom_kept(_op(), monkeypatch), "OOM raised")


def check_bound_by_buckets(op) -> None:
    """Calls over 28 token counts from 300 to 2919 bind one entry per token bucket
    (512, 1024, 2048, 4096), not one per count."""
    op.bind("kernel")
    rows = range(300, 3000, 97)
    for n in rows:
        op(torch.randn(n, 32))
    buckets = {choice.pow2(n, choice.TOKEN_FLOOR) for n in rows}
    assert len(op.bound) <= len(buckets), (len(op.bound), sorted(buckets))


def _exact_key(args) -> tuple:
    """The broken variant: every exact shape is its own binding."""
    return tuple(tuple(a.shape) if isinstance(a, torch.Tensor) else a for a in args)


def test_binding_cache_stays_bounded(fresh, monkeypatch):
    check_bound_by_buckets(_op())
    monkeypatch.setattr(choice, "_call_key", _exact_key)
    must_fail(lambda: check_bound_by_buckets(_op()), "one binding per token count")


def test_rebuild_in_target_dtype():
    t = choice._rebuild(("float", (8, 4), torch.bfloat16, torch.device("cpu"), True))
    assert t.dtype == torch.bfloat16 and t.requires_grad and t.shape == (8, 4)
