"""Tuning never moves the caller's random generators: a measuring select, the attention
prewarm, a replay, a kernel-vs-compile retune and the adapter prewarm draw their
synthetic inputs under `rng_kept`, and a cold tile tune of the installed LyCORIS
(its device probe included) draws nothing; the CPU (and CUDA) generator state after
them is the state before."""

import contextlib

import pytest
import torch
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.ops import get_ops
from lycoris.kernels.plans import tune as lycoris_tune
from lycoris.kernels.tuner import model as lycoris_model
from lycoris.kernels.tuner import tuner as lycoris_tuner
from lycoris.modules import LoConModule
from torch import nn

from kohakumoss.adapters.bind import autotuned, bodies, ops
from kohakumoss.adapters.lycoris.backend import pick_backend
from kohakumoss.adapters.lycoris.tuning import (
    restore_lycoris_tuning,
    set_lycoris_tuning,
)
from kohakumoss_kernels import recording, registry, status
from kohakumoss_kernels.backends.compile import choice
from kohakumoss_kernels.backends.eager import ffn
from kohakumoss_kernels.tuning import prewarm, replay, tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OP = "test_rng_double"


def _states():
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return torch.get_rng_state(), cuda


def check_kept(run) -> None:
    torch.manual_seed(1234)
    before = _states()
    run()
    after = _states()
    assert torch.equal(before[0], after[0]), "the CPU generator moved"
    assert all(torch.equal(a, b) for a, b in zip(before[1], after[1])), "CUDA moved"


def _unkept(monkeypatch, module) -> None:
    """The broken variant: the module draws on the caller's generators."""
    monkeypatch.setattr(module, "rng_kept", lambda devices=(): contextlib.nullcontext())


@pytest.fixture
def tuning(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})

    def unit(fn, dev):
        fn()
        return 1.0

    monkeypatch.setattr(tuner, "TIMINGS", {"events": unit, "burst": unit})
    tuner.reset()
    tuner.set_mode("table")
    yield
    tuner.reset()


def _measure():
    def launch(config):
        return lambda: torch.randn(64)

    tuner.select("rng_kernel", torch.device("cpu"), "k", lambda: [{"a": 1}, {}], launch)


def test_select_keeps_the_rng(tuning):
    check_kept(_measure)


def test_negative_select_without_fork(tuning, monkeypatch):
    _unkept(monkeypatch, tuner)
    must_fail(lambda: check_kept(_measure), "select drew on the caller's generator")


def _attention():
    def spy(q, k, v, cu, length, scale):
        return q

    prewarm.prewarm_attention(spy, "cpu", torch.float32, 1, 8, 128, backward=False)


def test_attention_prewarm_keeps_the_rng():
    check_kept(_attention)


def test_negative_attention_prewarm_without_fork(monkeypatch):
    _unkept(monkeypatch, prewarm)
    must_fail(lambda: check_kept(_attention), "attention prewarm drew")


def _retune():
    op = choice.kernel_or_compile("rng", "situ_glu", ffn.situ_glu, choice.shape_key)
    op.impls["compile"] = ffn.situ_glu
    op(torch.zeros(32, 16))
    assert op.pending
    choice.retune_pending()


def test_retune_keeps_the_rng(tuning):
    check_kept(_retune)


def test_negative_retune_without_fork(tuning, monkeypatch):
    _unkept(monkeypatch, choice)
    must_fail(lambda: check_kept(_retune), "retune drew")


@pytest.fixture
def recorded(tuning, monkeypatch):
    """A recorded call of `OP` (eager only) whose input is too large to keep, so the
    replay draws it."""
    monkeypatch.setattr(status, "_CHECKS", {})
    registry.register(OP, "eager", lambda x: x * 2)
    order = registry._STATE.priority["cpu"]
    op = registry.resolve(OP, "cpu", torch.float32)
    with recording.recording() as calls:
        op(torch.zeros(recording.KEEP_NUMEL + 1))
    yield calls
    registry.set_priority("cpu", order)
    registry._STATE.impls.pop(OP, None)


def _adapter_op():
    from_lyco = LoConModule("l", nn.Linear(64, 48), 1.0, 4, 4.0)
    return autotuned.AutotunedOp(
        ops.lora_bypass_with_base(from_lyco, 0.5), "triton", "tokens"
    )


def _adapter_warm(op):
    op.candidates = {name: _body for name in op.candidates}
    return lambda: op.warm([300], torch.float32, torch.device("cpu"))


def _body(operands):
    return bodies.lora_with_base(operands, 0.5)


def test_adapter_prewarm_keeps_the_rng(tuning, monkeypatch):
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    check_kept(_adapter_warm(_adapter_op()))


def test_negative_adapter_prewarm_without_fork(tuning, monkeypatch):
    monkeypatch.setattr(autotuned, "fusable", lambda operands: True)
    _unkept(monkeypatch, autotuned)
    must_fail(lambda: check_kept(_adapter_warm(_adapter_op())), "adapter warm drew")


@pytest.fixture
def lycoris_cold(device, tmp_path, monkeypatch):
    """The installed LyCORIS with an empty tile table and no device probe yet, tuning on
    through `set_lycoris_tuning`; yields the fused op set `pick_backend` takes."""
    if device.type != "cuda":
        pytest.skip("LyCORIS fused kernels need CUDA")
    backend = pick_backend(device)
    if backend not in fused_backends(device):
        pytest.skip(f"no LyCORIS fused backend on {device} ({backend})")
    monkeypatch.setenv("LYCORIS_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(lycoris_tune, "_CACHE", {})
    monkeypatch.setattr(lycoris_tune, "_UNTIMED", {})
    monkeypatch.setattr(lycoris_tune, "_LOADED", False)
    for name in ("_chosen", "_untimed", "_disk"):
        monkeypatch.setattr(lycoris_tuner, name, {})
    monkeypatch.setattr(lycoris_tuner, "record", [])
    lycoris_model._measure.cache_clear()
    previous = set_lycoris_tuning(True)
    yield get_ops(backend)
    restore_lycoris_tuning(previous)


def _lycoris_tune(ops, device):
    """A cold LyCORIS tile tune: the inputs drawn first, then one bypass and one Tucker
    forward on unseen shapes, each timing its configs."""
    draw = lambda *s: torch.randn(*s, device=device, dtype=torch.float16)  # noqa: E731
    x, up, down = draw(200, 96), draw(80, 8), draw(8, 96)
    a, t, b = draw(72, 8), draw(8, 8, 9), draw(8, 56)

    def run():
        ops.lora_bypass_fwd(x, up, down, gamma=0.5)
        ops.lora_tucker_fwd(a, t, b, gamma=0.5)
        torch.cuda.synchronize(device)

    return run


@pytest.mark.cuda
def test_lycoris_cold_tune_keeps_the_rng(lycoris_cold, device, monkeypatch):
    timed = []
    plan_time = lycoris_tune._time
    monkeypatch.setattr(
        lycoris_tune, "_time", lambda *a, **k: (timed.append(1), plan_time(*a, **k))[1]
    )
    check_kept(_lycoris_tune(lycoris_cold, device))
    assert lycoris_model._measure.cache_info().currsize == 1, "the probe did not run"
    assert lycoris_tuner.record or timed, "nothing was timed"


@pytest.mark.cuda
def test_negative_lycoris_probe_on_the_global_generator(
    lycoris_cold, device, monkeypatch
):
    """The broken variant: the probe's private generator dropped, so it draws on the
    caller's."""
    run = _lycoris_tune(lycoris_cold, device)
    randn = torch.randn

    def shared(*args, generator=None, **kwargs):
        return randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", shared)
    must_fail(lambda: check_kept(run), "the LyCORIS probe drew")


def test_replay_keeps_the_rng(recorded):
    check_kept(lambda: replay.replay(recorded, [{}], backward=False))


def test_negative_replay_without_fork(recorded, monkeypatch):
    _unkept(monkeypatch, replay)
    run = lambda: replay.replay(recorded, [{}], backward=False)  # noqa: E731
    must_fail(lambda: check_kept(run), "replay drew")
