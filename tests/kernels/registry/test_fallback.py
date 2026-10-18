"""Registry fallback: a backend that cannot run an op hands it to the next backend."""

import logging
import warnings

import pytest
import torch

from kohakumoss_kernels import registry, status
from kohakumoss_kernels.backends.compile import choice
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.backends.eager import ffn
from kohakumoss_kernels.errors import (
    BackendUnavailable,
    ConfigsFailed,
    InputError,
    TritonError,
    is_sticky_error,
    is_toolchain_error,
)
from kohakumoss_kernels.tuning import tuner
from tests._util.logs import collecting
from tests._util.negative import must_fail

OP = "test_fallback_double"
COMPILER_ERROR = "Failed to find C++ compiler. Please specify via CXX"


def _warnings(logs) -> list[str]:
    return [r.getMessage() for r in logs.records if r.levelno >= logging.WARNING]


def _key(backend: str, problem: str | None = None) -> str:
    """The failure key of `OP` on `backend` (cpu, float32), of one problem or of all."""
    return status.failure_key(OP, backend, "cpu", torch.float32, (), problem)


KERNEL_PROBLEM = f"{OP}_kernel:k"


def _no_compiler_kernel(calls):
    """A tuned kernel whose every config fails to build (no host compiler)."""

    def launch(config):
        def run():
            raise RuntimeError(f"{COMPILER_ERROR} (config {config['block']})")

        return run

    def kernel(x):
        calls["triton"] += 1
        tuner.select(
            f"{OP}_kernel", x.device, "k", lambda: [{"block": 1}, {"block": 2}], launch
        )
        return x * 2

    return kernel


@pytest.fixture
def broken_backend(tmp_path, monkeypatch):
    """`OP` on cpu with priority triton -> compile -> eager; triton cannot build; no
    backend probes (the fake backends are available); TUNE="table", so the tuned
    kernel times (and fails) its configs."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(status, "_CHECKS", {})
    status.clear_failures()
    tuner._TABLES.clear()
    tuner.reset()
    mode = tuner.get_mode()
    tuner.set_mode("table")
    calls = {"triton": 0, "compile": 0}

    def compiled(x):
        calls["compile"] += 1
        return x * 2

    registry.register(OP, "triton", _no_compiler_kernel(calls))
    registry.register(OP, "compile", compiled)
    registry.register(OP, "eager", lambda x: x * 2)
    order = registry._STATE.priority["cpu"]
    registry.set_priority("cpu", ("triton", "compile", "eager"))
    yield calls
    tuner.set_mode(mode)
    registry.set_priority("cpu", order)
    registry._STATE.impls.pop(OP, None)
    status.clear_failures()
    tuner._TABLES.clear()
    tuner.reset()


def check_fallback(fn, calls):
    """The op falls back to compile with a WARNING naming the backends and the error,
    records the failure of that one problem in this process only (a live call is never
    written to disk), and stops calling the failed backend; the backend still serves
    the op's other problems."""
    x = torch.arange(4.0)
    with collecting(logging.WARNING) as logs:
        try:
            out = fn(x)
        except BackendUnavailable as err:
            raise AssertionError(f"no fallback: {err}") from err
    assert torch.equal(out, x * 2)
    assert getattr(fn, "backend", None) == "compile"
    assert any(
        "backend triton failed, using compile" in m and COMPILER_ERROR in m
        for m in _warnings(logs)
    ), _warnings(logs)
    key = _key("triton", KERNEL_PROBLEM)
    assert COMPILER_ERROR in registry.failures()[key]
    assert key not in registry.recorded_failures(), "a live failure was persisted"
    fn(x)
    assert calls == {"triton": 1, "compile": 2}
    assert registry.available(OP, "cpu", torch.float32)["triton"] is None


def test_all_configs_fail_falls_back(broken_backend):
    check_fallback(registry.resolve(OP, "cpu", torch.float32), broken_backend)


def test_negative_without_fallback(broken_backend):
    """The raw implementation (no Bound) raises instead: the check must catch it."""
    raw = registry.resolve(OP, "cpu", torch.float32).fn
    must_fail(lambda: check_fallback(raw, broken_backend), "no registry fallback")


def test_pinned_backend_raises(broken_backend):
    """A pinned backend has nothing to fall back to: the tuner's error surfaces."""
    fn = registry.resolve(OP, "cpu", torch.float32, backend="triton")
    with pytest.raises(ConfigsFailed, match="every shortlisted config failed"):
        fn(torch.arange(4.0))


def _raise(x):
    raise RuntimeError(COMPILER_ERROR)


def test_kernel_or_compile_that_cannot_run_falls_back(broken_backend):
    """A kernel-vs-compile entry failing its device probe hands the op to the next
    backend."""
    registry.register(
        OP,
        "triton",
        choice.kernel_or_compile("test", "situ_glu", _raise, choice.shape_key),
    )
    registry.register(OP, "compile", ffn.situ_glu)
    fn = registry.resolve(OP, "cpu", torch.float32)
    x = torch.randn(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.testing.assert_close(fn(x), ffn.situ_glu(x))
    assert fn.backend == "compile"
    assert COMPILER_ERROR in registry.failures()[_key("triton")]


def _no_compiler_backend(gm, example_inputs):
    """A torch.compile backend that cannot build anything."""
    raise RuntimeError(COMPILER_ERROR)


def check_compile_fallback(fn):
    """The op runs the eager reference after its first compile fails, and records why."""
    x = torch.randn(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = fn(x)
        except BackendUnavailable as err:
            raise AssertionError(f"no fallback: {err}") from err
    torch.testing.assert_close(out, ffn.situ_glu(x))
    assert getattr(fn, "backend", None) == "eager"
    assert COMPILER_ERROR in registry.failures()[_key("compile")]


def _failing_compile_op():
    torch._dynamo.reset()
    registry.register(
        OP, "compile", compiled_op("situ_glu", True, _no_compiler_backend)
    )
    registry.register(OP, "eager", ffn.situ_glu)
    registry.set_priority("cpu", ("compile", "eager"))
    return registry.resolve(OP, "cpu", torch.float32)


def test_failing_compile_falls_back_to_eager(broken_backend):
    check_compile_fallback(_failing_compile_op())


def test_negative_failing_compile_without_fallback(broken_backend):
    raw = _failing_compile_op().fn
    must_fail(lambda: check_compile_fallback(raw), "no compile fallback")


def _checked_call(fn, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(x)
        except (BackendUnavailable, *([TritonError] if TritonError else [])) as err:
            raise AssertionError(f"no fallback: {err}") from err


@pytest.mark.skipif(TritonError is None, reason="triton not installed")
def test_tune_off_untimed_pick_build_error_falls_back(broken_backend):
    """TUNE=off launches the first candidate untimed; its build error (a Triton
    CompilationError / OutOfResources) still hands the op to the next backend."""
    tuner.set_mode("off")

    def untimed_kernel(x):
        raise TritonError("CompilationError: out of resources: shared memory")

    registry.register(OP, "triton", untimed_kernel)
    fn = registry.resolve(OP, "cpu", torch.float32)
    x = torch.arange(4.0)
    assert torch.equal(_checked_call(fn, x), x * 2)
    assert fn.backend == "compile"
    must_fail(lambda: _checked_call(untimed_kernel, x), "raw kernel, no Bound")


def test_unavailable_backend_is_never_bound(broken_backend, monkeypatch):
    """A backend whose smoke build fails on the device is skipped by resolve, with
    the reason in `available` (the doctor's view)."""

    def no_compiler(device):
        raise RuntimeError(COMPILER_ERROR)

    monkeypatch.setitem(status._CHECKS, "triton", no_compiler)
    with collecting(logging.WARNING) as logs:
        fn = registry.resolve(OP, "cpu", torch.float32)
    assert fn.backend == "compile"
    assert any(COMPILER_ERROR in m for m in _warnings(logs))
    assert "cannot build" in registry.available(OP, "cpu", torch.float32)["triton"]
    monkeypatch.setattr(status, "_CHECKS", {})
    status.clear_failures()
    must_fail(
        lambda: _assert(registry.resolve(OP, "cpu", torch.float32).backend == "compile")
    )


def _assert(cond):
    assert cond


class _BadBackward(torch.autograd.Function):
    """Forward fine, backward kernel cannot build."""

    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, g):
        raise ConfigsFailed("bwd_kernel", {"cfg": COMPILER_ERROR})


def _fwd_bwd(op):
    x = torch.arange(4.0, requires_grad=True)
    op(x).sum().backward()
    return x.grad


def test_backward_only_failure_falls_back_in_prewarm(broken_backend):
    """A backward-only build failure met while exercising the op (prewarm) falls the
    op back before training; the grads come from the next backend."""
    registry.register(OP, "triton", _BadBackward.apply)
    fn = registry.resolve(OP, "cpu", torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = registry.exercise(fn, _fwd_bwd)
    assert torch.equal(grad, torch.full((4,), 2.0))
    assert fn.backend == "compile"
    assert COMPILER_ERROR in registry.failures()[_key("triton")]
    fresh = registry.resolve(OP, "cpu", torch.float32)
    assert fresh.backend == "compile", "a recorded failure skips the backend"
    registry.register(OP, "triton", _BadBackward.apply)
    status.clear_failures()
    must_fail(lambda: _assert_raises_none(lambda: _fwd_bwd(_unguarded())))


def _unguarded():
    """The resolved op with the backward guard off (the broken variant)."""
    op = registry.resolve(OP, "cpu", torch.float32)
    op.guard = False
    return op


def check_live_backward_fallback(op) -> None:
    """A backward-only failure in a live step (no prewarm) falls back inside
    loss.backward(): correct gradient, next backend bound, failure recorded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grad = _assert_raises_none(lambda: _fwd_bwd(op))
    assert torch.equal(grad, torch.full((4,), 2.0))
    assert op.backend == "compile"
    assert COMPILER_ERROR in registry.failures()[_key("triton")]


def test_live_backward_only_failure_falls_back(broken_backend):
    registry.register(OP, "triton", _BadBackward.apply)
    check_live_backward_fallback(registry.resolve(OP, "cpu", torch.float32))


def test_negative_live_backward_without_guard(broken_backend):
    registry.register(OP, "triton", _BadBackward.apply)
    must_fail(lambda: check_live_backward_fallback(_unguarded()), "no guard")


def test_guard_drops_after_a_good_backward(broken_backend):
    """After one successful backward the op stays guarded to the step's end
    (`settle_guards`), then calls its implementation directly."""
    registry.register(OP, "triton", lambda x: x * 2)
    fn = registry.resolve(OP, "cpu", torch.float32)
    assert fn.guard
    _fwd_bwd(fn)
    assert fn.guard, "the call path changed inside the step"
    assert registry.settle_guards() >= 1
    assert not fn.guard and fn.backend == "triton"


def _assert_raises_none(fn):
    try:
        return fn()
    except BackendUnavailable as err:
        raise AssertionError(f"raised {err}") from err


def test_out_of_memory_never_falls_back(broken_backend):
    """Out-of-memory in a live launch propagates and records nothing; while tuning it
    skips the candidate (the default binds for this process, nothing stored)."""

    def oom(x):
        raise torch.OutOfMemoryError("CUDA out of memory")

    registry.register(OP, "triton", oom)
    fn = registry.resolve(OP, "cpu", torch.float32)
    with pytest.raises(torch.OutOfMemoryError):
        fn(torch.arange(4.0))
    assert fn.backend == "triton" and not registry.failures()

    def launch(config):
        def run():
            raise torch.OutOfMemoryError("CUDA out of memory while tuning")

        return run

    cpu = torch.device("cpu")
    got = tuner.select("oom_kernel", cpu, "k", lambda: [{}, {"a": 1}], launch)
    assert got == {} and not tuner.stored("oom_kernel", cpu, "k")
    assert fn.backend == "triton" and not registry.failures()


def _prewarm(fn, x):
    """The op exercised at build time (`registry.exercise`, as the prewarm runs it)."""
    return registry.exercise(fn, lambda op: _checked_call(op, x))


def test_recorded_failure_is_honoured_by_a_new_process(broken_backend):
    """A whole-op build failure met at build time is on disk under the current
    fingerprint and skips the backend in a fresh process; under another fingerprint it
    is ignored."""
    registry.register(OP, "triton", _raise_unavailable)
    fn = registry.resolve(OP, "cpu", torch.float32)
    _prewarm(fn, torch.arange(4.0))
    status._FAILED.clear()
    status._DISK.clear()
    status._LOADED.clear()
    assert registry.resolve(OP, "cpu", torch.float32).backend == "compile"
    for row in status._disk().values():
        row["fingerprint"] = "other"
    assert registry.resolve(OP, "cpu", torch.float32).backend == "triton"


def check_failure_scope(resolve_after) -> None:
    """A failure of one problem leaves the backend for the op's other problems; a
    whole-op failure for float32 leaves it for bfloat16."""
    fn = registry.resolve(OP, "cpu", torch.float32)
    _prewarm(fn, torch.arange(4.0))
    assert resolve_after(torch.float32) == "triton", "one problem disabled the op"
    whole_op = BackendUnavailable(OP, {"x": COMPILER_ERROR})
    with status.build_phase():
        status.record(OP, "triton", "cpu", whole_op, torch.float32)
    assert resolve_after(torch.float32) == "compile"
    assert resolve_after(torch.bfloat16) == "triton", "a float32 failure hit bfloat16"


def _fresh_backend(dtype) -> str:
    """The backend a later process resolves (memory cleared, disk kept)."""
    status._FAILED.clear()
    status._DISK.clear()
    status._LOADED.clear()
    return registry.resolve(OP, "cpu", dtype).backend


def test_failure_scope(broken_backend):
    check_failure_scope(_fresh_backend)


def test_negative_failure_key_without_scope(broken_backend, monkeypatch):
    monkeypatch.setattr(
        status, "failure_key", lambda op, backend, arch, *rest: f"{op}|{backend}|{arch}"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        must_fail(lambda: check_failure_scope(_fresh_backend), "op|backend|arch key")


def test_fallback_inside_a_compiled_region(broken_backend):
    """A Bound op called inside torch.compile still falls back when its backend
    cannot build (dynamo runs the fallback eagerly)."""
    registry.register(OP, "triton", _raise_unavailable)
    fn = registry.resolve(OP, "cpu", torch.float32)
    torch._dynamo.reset()
    compiled = torch.compile(lambda x: fn(x) + 1, backend="eager")
    x = torch.arange(4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert torch.equal(compiled(x), x * 2 + 1)
    assert fn.backend == "compile"


def _raise_unavailable(x):
    raise BackendUnavailable(OP, {"cpu": COMPILER_ERROR})


def _pointer_kernel(message):
    """A tuned two-argument kernel whose every config raises `message` (ValueError)."""

    def launch(config):
        def run():
            raise ValueError(f"{message} (config {config['block']})")

        return run

    def kernel(x, w):
        tuner.select(
            f"{OP}_kernel",
            x.device,
            f"k{tuple(x.shape)}{tuple(w.shape)}",
            lambda: [{"block": 1}, {"block": 2}],
            launch,
            errors=(RuntimeError, ValueError),
        )
        return x * w

    return kernel


def _two_arg_op(message):
    registry.register(OP, "triton", _pointer_kernel(message))
    registry.register(OP, "compile", lambda x, w: x * w)
    registry.register(OP, "eager", lambda x, w: x * w)
    return registry.resolve(OP, "cpu", torch.float32)


def check_input_error_propagates(op) -> None:
    """Arguments the eager reference rejects (shape mismatch) raise `InputError`; the op
    stays bound and no failure is recorded."""
    try:
        op(torch.ones(4), torch.ones(3))
    except InputError as err:
        assert "eager reference also rejects" in str(err)
    except Exception as err:
        raise AssertionError(f"not an InputError: {err!r}") from err
    else:
        raise AssertionError("bad arguments raised nothing")
    assert op.backend == "triton", op.backend
    assert not registry.failures() and not registry.recorded_failures()


def test_input_error_never_demotes(broken_backend):
    check_input_error_propagates(_two_arg_op("Pointer argument (cpu tensor?)"))


def test_negative_fallback_without_input_check(broken_backend, monkeypatch):
    op = _two_arg_op("Pointer argument (cpu tensor?)")
    monkeypatch.setattr(registry.Bound, "_check_inputs", lambda *a: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        must_fail(lambda: check_input_error_propagates(op), "no input check")


def check_launch_failure_not_persisted(op, prewarm: bool) -> None:
    """A backend-only failure on valid arguments falls back in this process but is not
    written to disk (it is not a build / toolchain cause), not even at build time."""
    args = (torch.ones(4), torch.full((4,), 3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if prewarm:
            out = registry.exercise(op, lambda o: o(*args))
        else:
            out = op(*args)
    assert torch.equal(out, torch.full((4,), 3.0)) and op.backend == "compile"
    key = _key("triton", f"{OP}_kernel:k(4,)(4,)")
    assert key in registry.failures(), registry.failures()
    assert key not in registry.recorded_failures()


@pytest.mark.parametrize("prewarm", [False, True])
def test_launch_failure_is_not_persisted(broken_backend, prewarm):
    check_launch_failure_not_persisted(_two_arg_op("unexpected stride"), prewarm)


def test_negative_every_failure_persisted(broken_backend, monkeypatch):
    op = _two_arg_op("unexpected stride")
    monkeypatch.setattr(status, "is_toolchain_error", lambda err: True)
    must_fail(
        lambda: check_launch_failure_not_persisted(op, True), "launch failure on disk"
    )


def check_live_toolchain_failure_kept_in_memory(fn) -> None:
    """A build failure first met in a live call (after the prewarm) rebinds the op in
    this process and is never written to disk."""
    _checked_call(fn, torch.arange(4.0))
    assert fn.backend == "compile"
    assert _key("triton") in registry.failures()
    assert not registry.recorded_failures(), registry.recorded_failures()


def test_live_toolchain_failure_not_persisted(broken_backend):
    registry.register(OP, "triton", _raise_unavailable)
    check_live_toolchain_failure_kept_in_memory(
        registry.resolve(OP, "cpu", torch.float32)
    )


def test_negative_live_failure_persisted(broken_backend, monkeypatch):
    registry.register(OP, "triton", _raise_unavailable)
    monkeypatch.setattr(status, "_BUILD", [True])
    fn = registry.resolve(OP, "cpu", torch.float32)
    must_fail(lambda: check_live_toolchain_failure_kept_in_memory(fn), "live on disk")


def check_siblings_rebind(twin, model) -> None:
    """The prewarm exercising the twin's op moves the model's op of the same query off
    the failing backend too, before the model's first call."""
    _prewarm(twin, torch.arange(4.0))
    assert twin.backend == "compile"
    assert model.backend == "compile", "the model's op stayed on the failed backend"


def test_prewarm_failure_rebinds_the_model_op(broken_backend):
    registry.register(OP, "triton", _raise_unavailable)
    twin = registry.resolve(OP, "cpu", torch.float32)
    check_siblings_rebind(twin, registry.resolve(OP, "cpu", torch.float32))


def test_negative_twin_only(broken_backend, monkeypatch):
    registry.register(OP, "triton", _raise_unavailable)
    twin = registry.resolve(OP, "cpu", torch.float32)
    model = registry.resolve(OP, "cpu", torch.float32)
    monkeypatch.setattr(registry, "_LIVE", set())
    must_fail(lambda: check_siblings_rebind(twin, model), "only the twin rebinds")


STICKY = "CUDA error: an illegal memory access was encountered"


def check_sticky_reraised(fn) -> None:
    """A sticky CUDA error propagates and leaves the op bound (no fallback onto a
    broken context); a launch error on a sound context still falls back."""
    try:
        fn(torch.arange(4.0))
    except BackendUnavailable as err:
        assert "illegal memory access" in str(err), err
    else:
        raise AssertionError("the sticky error did not propagate")
    assert fn.backend == "triton", fn.backend
    assert not registry.failures(), registry.failures()


def _sticky(x):
    """A kernel entry translating a sticky launch error (`errors.unavailable`)."""
    raise BackendUnavailable(OP, {"cpu": f"RuntimeError: {STICKY}"})


def test_sticky_error_never_falls_back(broken_backend):
    registry.register(OP, "triton", _sticky)
    check_sticky_reraised(registry.resolve(OP, "cpu", torch.float32))
    assert not is_sticky_error(RuntimeError(COMPILER_ERROR))
    assert is_sticky_error(RuntimeError("device-side assert triggered"))


def test_negative_sticky_error_falls_back(broken_backend, monkeypatch):
    registry.register(OP, "triton", _sticky)
    monkeypatch.setattr(registry, "is_sticky_error", lambda err, device=None: False)
    fn = registry.resolve(OP, "cpu", torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        must_fail(lambda: check_sticky_reraised(fn), "fallback onto a broken context")


def test_toolchain_classes():
    assert is_toolchain_error(RuntimeError(COMPILER_ERROR))
    assert is_toolchain_error(OSError("cannot load nvrtc"))
    assert not is_toolchain_error(ValueError("Pointer argument (cpu tensor?)"))
    assert not is_toolchain_error(
        RuntimeError("Expected all tensors to be on the same device")
    )
    assert not is_toolchain_error(torch.OutOfMemoryError("CUDA out of memory"))
    assert not is_toolchain_error(ConfigsFailed("k", {"a": "x"}, persist=False))
