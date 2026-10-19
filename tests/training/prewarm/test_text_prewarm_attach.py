"""The text-encoder prewarm runs only when the run encodes text, model-free: attached under
TUNE="table" for an online / uncached text encoder on an accelerator, it runs once,
right before the encoder's first pass; the step-1 VRAM window and step-1 speed leave
a prewarm inside step 1 out."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.encoders.text.encoder import QwenTextEncoder
from kohakumoss.training.build import kernels
from kohakumoss.training.callbacks import kernels as step_kernels
from kohakumoss.training.state import TrainState
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.hardware import memory
from tests._util.memory import read_window
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable
from tests.training.memory.test_vram_accounting import FakeMem, gib
from tests.training.prewarm.test_prewarm_shapes import _setup

pytestmark = pytest.mark.cpu
CUDA = torch.device("cuda", 0)
CPU = torch.device("cpu")


def _encoder(log: list) -> QwenTextEncoder:
    """An encoder shell whose pass only logs (no model)."""
    te = object.__new__(QwenTextEncoder)
    te._run = lambda enc, prompts: log.append(("pass", tuple(prompts))) or []
    return te


def check_once_before(attach) -> None:
    """`attach(te, fn)` runs `fn` once, before the first of two passes."""
    log = []
    te = _encoder(log)
    attach(te, lambda: log.append("warm"))
    te.encode(["a"])
    te.encode(["b"])
    assert log == ["warm", ("pass", ("a",)), ("pass", ("b",))], log


def _after_each(te, fn):
    run = te._run

    def each(enc, prompts):
        out = run(enc, prompts)
        fn()
        return out

    te._run = each


def test_before_first_runs_once_before_the_pass():
    check_once_before(QwenTextEncoder.before_first)
    must_fail(lambda: check_once_before(_after_each), "after every pass")


class _Spy:
    def __init__(self, device):
        self.device = device
        self.attached = []

    def before_first(self, fn):
        self.attached.append(fn)


def check_attached(k: dict, device=CUDA) -> None:
    te = _Spy(device)
    data = SimpleNamespace(text_tokens_max=lambda: 300)
    kernels.attach_text_prewarm(k, SimpleNamespace(text_encoder=te), data)
    assert len(te.attached) == 1, te.attached
    (warm,) = te.attached
    assert warm.data_tokens == 300 and warm.max_text_tokens == k["MAX_TEXT_TOKENS"]


def test_armed_only_when_text_is_encoded_in_the_loop():
    for cache in ("online", "off"):
        check_attached(runnable(TUNE="table", TEXT_CACHE=cache))
    must_fail(lambda: check_attached(runnable(TUNE="table", TEXT_CACHE="offline")))
    must_fail(lambda: check_attached(runnable(TUNE="off", TEXT_CACHE="online")))
    k = runnable(TUNE="table", TEXT_CACHE="online")
    must_fail(lambda: check_attached(k, torch.device("cpu")), "the host")


@pytest.fixture
def fake(monkeypatch):
    mem = FakeMem()
    monkeypatch.setattr(memory, "MODULES", {"cuda": mem})
    monkeypatch.setattr(memory, "_OPEN", {})
    return mem


def check_step_one(fake, late: bool) -> None:
    """A 10 GiB prewarm at the start of step 1, then a 4 GiB step: step 1 reports 4 GiB
    and 1 s."""
    ticks = iter([0.0, 5.0, 6.0])
    speed = step_kernels.StartupSpeed(last=2, clock=lambda: next(ticks))
    vram = step_kernels.PrewarmVram(CUDA)
    fake.set(10.0)
    fake.set(2.0)
    if late:
        step_kernels.late_prewarm(CUDA, 4.0)
    fake.set(4.0)
    fake.set(2.0)
    step_alloc = read_window(vram.window)[0]
    vram.on_step_end(TrainState(), None)
    speed.on_step_end(None, None)
    assert step_alloc == gib(4.0), step_alloc / gib(1.0)
    assert speed.marks[1] - speed.marks[0] == 1.0, speed.marks


def test_step_one_leaves_a_late_prewarm_out(fake):
    check_step_one(fake, True)
    must_fail(lambda: check_step_one(fake, False), "the prewarm counted in step 1")


def _report(gib_peak: float) -> dict:
    return {
        "op": SimpleNamespace(peak_alloc=gib(gib_peak), peak_reserved=gib(gib_peak))
    }


def check_run_peak(fake) -> None:
    """A second run in the process reports its own prewarm peak (11.5 GiB), not the
    first run's (24.6 GiB)."""
    k, prec, real = _setup()
    built = SimpleNamespace(backbone=real, plan=MemoryPlan())
    kernels.note_peak(CUDA, _report(24.6))
    kernels.build_prewarm(k, built, prec, CPU)
    kernels.note_peak(CUDA, _report(11.5))
    state = TrainState()
    step_kernels.PrewarmVram(CUDA).on_step_end(state, None)
    peak = state.metrics["prewarm_peak_alloc"]
    assert peak == pytest.approx(11.5, abs=1e-6), peak


def test_prewarm_peak_is_per_run(fake, monkeypatch):
    check_run_peak(fake)
    monkeypatch.setattr(kernels, "reset_peaks", lambda: None)
    must_fail(lambda: check_run_peak(fake), "a process-wide max")
