"""Registry fallback: a backend that cannot run an op hands it to the next backend."""

import warnings

import pytest
import torch

from kohakumoss_kernels import health, registry
from kohakumoss_kernels.backends.compile import race
from kohakumoss_kernels.backends.compile.ops import compiled_op
from kohakumoss_kernels.backends.eager import ffn
from kohakumoss_kernels.errors import BackendUnavailable, ConfigsFailed, TritonError
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

OP = "test_fallback_double"
COMPILER_ERROR = "Failed to find C++ compiler. Please specify via CXX"


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
    backend health checks (the fake backends are healthy)."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(health, "_CHECKS", {})
    health.clear_failures()
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()
    mode = tuner.get_mode()
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
    health.clear_failures()
    tuner._TABLES.clear()
    tuner._CHOSEN.clear()


def check_fallback(fn, calls):
    """The op falls back to compile with a warning naming the errors, records the
    failure, and stops calling the failed backend."""
    x = torch.arange(4.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = fn(x)
        except BackendUnavailable as err:
            raise AssertionError(f"no fallback: {err}") from err
    assert torch.equal(out, x * 2)
    assert getattr(fn, "backend", None) == "compile"
    assert any(
        "backend triton failed" in str(w.message) and COMPILER_ERROR in str(w.message)
        for w in caught
    )
    assert COMPILER_ERROR in registry.failures()[f"{OP}|triton|cpu"]
    assert COMPILER_ERROR in registry.recorded_failures()[f"{OP}|triton|cpu"]["error"]
    fn(x)
    assert calls == {"triton": 1, "compile": 2}
    again = registry.resolve(OP, "cpu", torch.float32)
    assert again.backend == "compile"
    assert "failed here" in registry.available(OP, "cpu", torch.float32)["triton"]


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


def test_raced_kernel_that_cannot_run_falls_back(broken_backend):
    """A raced kernel failing its device probe hands the op to the next backend."""
    registry.register(
        OP, "triton", race.raced("test", "situ_glu", _raise, race.shape_key)
    )
    registry.register(OP, "compile", ffn.situ_glu)
    fn = registry.resolve(OP, "cpu", torch.float32)
    x = torch.randn(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.testing.assert_close(fn(x), ffn.situ_glu(x))
    assert fn.backend == "compile"
    assert COMPILER_ERROR in registry.failures()[f"{OP}|triton|cpu"]


def _no_compiler_backend(gm, example_inputs):
    """A torch.compile backend that cannot build anything."""
    raise RuntimeError(COMPILER_ERROR)


def check_compile_fallback(fn):
    """The op runs the eager spec after its first compile fails, and records why."""
    x = torch.randn(8, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = fn(x)
        except BackendUnavailable as err:
            raise AssertionError(f"no fallback: {err}") from err
    torch.testing.assert_close(out, ffn.situ_glu(x))
    assert getattr(fn, "backend", None) == "eager"
    assert COMPILER_ERROR in registry.failures()[f"{OP}|compile|cpu"]


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


def test_unhealthy_backend_is_never_bound(broken_backend, monkeypatch):
    """A backend whose smoke build fails on the device is skipped by resolve, with
    the reason in `available` (the doctor's view)."""

    def no_compiler(device):
        raise RuntimeError(COMPILER_ERROR)

    monkeypatch.setitem(health._CHECKS, "triton", no_compiler)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = registry.resolve(OP, "cpu", torch.float32)
    assert fn.backend == "compile"
    assert any(COMPILER_ERROR in str(w.message) for w in caught)
    assert "cannot build" in registry.available(OP, "cpu", torch.float32)["triton"]
    monkeypatch.setattr(health, "_CHECKS", {})
    health.clear_failures()
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
    assert COMPILER_ERROR in registry.failures()[f"{OP}|triton|cpu"]
    fresh = registry.resolve(OP, "cpu", torch.float32)
    assert fresh.backend == "compile", "a recorded failure skips the backend"
    registry.register(OP, "triton", _BadBackward.apply)
    health.clear_failures()
    must_fail(
        lambda: _assert_raises_none(
            lambda: _fwd_bwd(registry.resolve(OP, "cpu", torch.float32))
        )
    )


def _assert_raises_none(fn):
    try:
        fn()
    except BackendUnavailable as err:
        raise AssertionError(f"raised {err}") from err


def test_out_of_memory_never_falls_back(broken_backend):
    """Out-of-memory in a launch or while tuning propagates and records nothing."""

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

    with pytest.raises(torch.OutOfMemoryError):
        tuner.select(
            "oom_kernel", torch.device("cpu"), "k", lambda: [{}, {"a": 1}], launch
        )


def test_recorded_failure_is_honoured_by_a_new_process(broken_backend):
    """A failure on disk under the current fingerprint skips the backend in a fresh
    process; under another fingerprint it is ignored."""
    fn = registry.resolve(OP, "cpu", torch.float32)
    _checked_call(fn, torch.arange(4.0))
    health._FAILED.clear()
    health._DISK.clear()
    health._LOADED.clear()
    assert registry.resolve(OP, "cpu", torch.float32).backend == "compile"
    for row in health._disk().values():
        row["fingerprint"] = "other"
    assert registry.resolve(OP, "cpu", torch.float32).backend == "triton"


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
