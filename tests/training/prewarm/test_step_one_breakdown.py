"""Step-1 breakdown: the deltas of the encode / tuning / compile counters over step 1,
the unexplained rest, the JSON record and the metrics; only step 1 is reported. The
stack sampler charges each stack to its most specific category."""

import json
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training.callbacks import sampler
from kohakumoss.training.callbacks.sampler import StackSampler, classify
from kohakumoss.training.callbacks.startup import StepOneBreakdown, attribution
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _run(tmp_path, monkeypatch, tuned: float):
    monkeypatch.setattr(tuner, "TIMED_SECS", {"k": 1.0})
    clock = Clock()
    text = SimpleNamespace(stats={"seconds": 2.0})
    vae = SimpleNamespace(stats={"seconds": 0.5})
    cb = StepOneBreakdown(
        text, vae, str(tmp_path), torch.device("cpu"), clock, sample=False
    )
    clock.t += 10.0
    text.stats["seconds"] += 3.0
    vae.stats["seconds"] += 1.0
    tuner.TIMED_SECS["k"] += tuned
    state = SimpleNamespace(metrics={})
    cb.on_step_end(state, None)
    cb.on_step_end(SimpleNamespace(metrics={}), None)
    path = tmp_path / "startup" / "step1_breakdown.json"
    return state, json.loads(path.read_text())["counters"]


def check_breakdown(state, record) -> None:
    assert record["wall"] == pytest.approx(10.0)
    assert record["text_encode"] == pytest.approx(3.0)
    assert record["vae_encode"] == pytest.approx(1.0)
    assert record["kernel_tuning"] == pytest.approx(2.0)
    rest = 10.0 - 3.0 - 1.0 - 2.0 - record["torch_compile"]
    assert record["other"] == pytest.approx(rest)
    assert state.metrics["step1_text_encode"] == pytest.approx(3.0)


def test_step_one_breakdown(tmp_path, monkeypatch):
    check_breakdown(*_run(tmp_path, monkeypatch, 2.0))


def test_negative_tuning_not_counted(tmp_path, monkeypatch):
    state, record = _run(tmp_path, monkeypatch, 0.0)
    must_fail(lambda: check_breakdown(state, record), "tuning seconds missed")


STACKS = {
    "kernel_tuning": [
        ("torch", "matmul"),
        ("kohakumoss_kernels.tuning.timing", "best_ms"),
        ("kohakumoss.adapters.bind.autotuned", "resolve"),
        ("kohakumoss.training.step", "__call__"),
    ],
    "triton_jit_load": [
        ("triton.compiler.compiler", "_init_handles"),
        ("triton.runtime.jit", "run"),
        ("kohakumoss.training.step", "__call__"),
    ],
    "forward_backward": [
        ("triton.runtime.jit", "run"),
        ("kohakumoss.training.step", "__call__"),
    ],
    "stream_wait": [
        ("kohakumoss.offload.weights.layers", "_wait_device"),
        ("kohakumoss.offload.weights.layers", "enter"),
        ("kohakumoss.training.step", "__call__"),
    ],
}


def check_classes(classify_fn) -> None:
    for want, stack in STACKS.items():
        assert classify_fn(stack) == want, (want, classify_fn(stack))


def test_stack_categories():
    check_classes(classify)


def test_negative_innermost_frame_only(monkeypatch):
    """Charging a stack by its module rules alone misses the stream wait around a
    plain launch."""
    monkeypatch.setattr(
        sampler, "CATEGORIES", tuple(c for c in sampler.CATEGORIES if c[2] is None)
    )
    must_fail(lambda: check_classes(classify), "module-only rules")


class Frame:
    """A stand-in frame: module name, function, caller."""

    def __init__(self, module: str, func: str, back=None):
        self.f_globals = {"__name__": module}
        self.f_code = SimpleNamespace(co_name=func)
        self.f_back = back


def _chain(stack: list[tuple[str, str]]) -> Frame:
    """The innermost frame of `stack` (innermost first)."""
    frame = None
    for module, func in reversed(stack):
        frame = Frame(module, func, frame)
    return frame


MAIN, LOADER = 1, 2
TIMELINE = (
    [{MAIN: STACKS["kernel_tuning"], LOADER: STACKS["stream_wait"]}] * 3
    + [{MAIN: [("tests.fake", "spin"), ("kohakumoss.offload.x", "other")]}] * 2
    + [{MAIN: STACKS["triton_jit_load"]}]
)


def _sampled(sampler_cls, timeline) -> dict:
    snaps = iter(timeline)
    names = {MAIN: "MainThread", LOADER: "loader"}
    s = sampler_cls(
        interval=0.01,
        frames=lambda: {i: _chain(st) for i, st in next(snaps).items()},
        name=names.__getitem__,
    )
    for _ in timeline:
        s.sample()
    return attribution(s.report(), "MainThread")


def check_attribution(got: dict) -> None:
    assert got["main"] == pytest.approx(
        {"kernel_tuning": 0.03, "other": 0.02, "triton_jit_load": 0.01}
    )
    assert got["other"] == pytest.approx({"tests.fake.spin": 0.02})
    assert got["threads"] == {"loader": pytest.approx({"stream_wait": 0.03})}


def test_sampler_attribution():
    check_attribution(_sampled(StackSampler, TIMELINE))


def test_negative_sampler_dropping_threads():
    """A sampler that charges only the main thread loses the loader's stream wait."""

    class MainOnly(StackSampler):
        def sample(self):
            frames = self.frames
            self.frames = lambda: {i: f for i, f in frames().items() if i == MAIN}
            try:
                super().sample()
            finally:
                self.frames = frames

    must_fail(lambda: check_attribution(_sampled(MainOnly, TIMELINE)), "main only")
