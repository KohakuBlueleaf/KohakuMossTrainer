"""TUNE, model-free: "table" binds stored decisions and times a missing key once (then
binds the stored winner); "off" binds a stored winner, else the first candidate, and
times nothing; inference defaults to "off" and an owned pipeline sets LyCORIS's tuning
from its TUNE, with no prewarm; training under "off" binds no prewarm or retune; the
prewarm and step-speed log lines exist. Real generates and training runs under each
value run in tests/integration."""

import inspect
from pathlib import Path

import pytest
import torch
from kohakuengine import introspect_schema
from lycoris.kernels.plans.tune import tuning_enabled

from kohakumoss.adapters.lycoris.tuning import (
    restore_lycoris_tuning,
    set_lycoris_tuning,
)
from kohakumoss.inference import MossPipeline, pipeline
from kohakumoss.model.opset import OpSet
from kohakumoss.training.build import kernels as train_kernels
from kohakumoss.training.callbacks import kernels as step_kernels
from kohakumoss_kernels.backends.compile.choice import retune_pending
from kohakumoss_kernels.tuning import tuner
from tests._util.blockops_prewarm import prewarm_block_ops
from tests._util.logs import assert_phases
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = [{"block": 64}, {"block": 128}]


@pytest.fixture
def timings(monkeypatch, tmp_path):
    """Count every timed measurement; a fresh table directory and tuner state."""
    calls = []
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    monkeypatch.setattr(tuner, "_TABLES", {})
    monkeypatch.setattr(tuner, "_CHOSEN", {})
    spy = {k: (lambda fn, device, _k=k: calls.append(_k) or 1.0) for k in tuner.TIMINGS}
    monkeypatch.setattr(tuner, "TIMINGS", spy)
    yield calls


def select(key: str = "k") -> dict:
    return tuner.select(
        "test_kernel",
        torch.device("cpu"),
        key,
        lambda: CANDIDATES,
        lambda config: (lambda: None),
    )


def _timed(calls: list, n: int) -> None:
    assert len(calls) == n, calls


def test_table_times_a_missing_key_once(timings):
    tuner.set_mode("table")
    select()
    _timed(timings, len(CANDIDATES))
    tuner._CHOSEN.clear()
    select()
    _timed(timings, len(CANDIDATES))
    tuner.table("test_kernel", torch.device("cpu")).put("k2", CANDIDATES[1], {}, {})
    assert select("k2") == CANDIDATES[1]
    _timed(timings, len(CANDIDATES))
    must_fail(
        lambda: (select("k3"), _timed(timings, len(CANDIDATES))),
        "a new key timed nothing",
    )


def _off_binds_stored(timings) -> None:
    tuner._CHOSEN.clear()
    tuner.set_mode("off")
    assert select() == CANDIDATES[1]
    assert select("k-new") == CANDIDATES[0]
    _timed(timings, 0)


def test_off_binds_a_stored_winner_untimed(timings, monkeypatch):
    tuner.table("test_kernel", torch.device("cpu")).put("k", CANDIDATES[1], {}, {})
    _off_binds_stored(timings)
    tuner.set_mode("table")
    tuner._CHOSEN.clear()
    assert select() == CANDIDATES[1]
    _timed(timings, 0)
    with monkeypatch.context() as patch:
        patch.setattr(tuner.TuneTable, "get", lambda self, key: None)
        must_fail(lambda: _off_binds_stored(timings), "off ignores the table")


def test_set_mode_takes_only_the_tune_values():
    for mode in tuner.MODES:
        tuner.set_mode(mode)
    with pytest.raises(ValueError):
        tuner.set_mode("auto")


def _pipeline_tune(tune: str, lycoris: bool, retunes: bool) -> None:
    pipe = MossPipeline(None, device="cpu", tune=tune)
    assert tuner.get_mode() == tune
    assert tuning_enabled() is lycoris
    assert (pipe.retune is retune_pending) == retunes
    assert not hasattr(pipe, "warm")


def test_inference_defaults_to_off_without_prewarm(monkeypatch):
    for fn in (MossPipeline.__init__, MossPipeline.from_pretrained):
        assert inspect.signature(fn).parameters["tune"].default == "off"
    for script in ("sample.py", "compare_grid.py"):
        defaults = introspect_schema(ROOT / "scripts" / "sample" / script).defaults
        assert defaults["TUNE"] == "off", script
    previous = set_lycoris_tuning(True)
    try:
        _pipeline_tune("off", False, False)
        _pipeline_tune("table", True, True)
        MossPipeline.from_components(None, device="cpu")
        assert tuner.get_mode() == "table"
        monkeypatch.setattr(pipeline, "set_lycoris_tuning", lambda on: None)
        must_fail(lambda: _pipeline_tune("off", False, False), "LyCORIS left as it was")
    finally:
        restore_lycoris_tuning(previous)


def _no_retune(names: list[str]) -> None:
    assert "RetuneKernels" not in names


def test_training_off_binds_no_tuning():
    k = runnable(TUNE="off")
    cpu = torch.device("cpu")
    assert train_kernels.build_prewarm(k, None, None, cpu)([cpu]) is None
    _no_retune([type(cb).__name__ for cb in step_kernels.retune_callbacks(k)])
    speed = [type(cb).__name__ for cb in step_kernels.speed_callbacks(k)]
    assert speed == ["StartupSpeed"]
    tuned = runnable(TUNE="table")
    tuned_names = [type(cb).__name__ for cb in step_kernels.retune_callbacks(tuned)]
    must_fail(lambda: _no_retune(tuned_names), "table binds no retune")


def test_prewarm_log_lines(collect):
    ops = OpSet.resolve("cpu", torch.float32, {})
    names = ("rms_norm_mod", "rms_norm", "gated_residual", "situ_glu", "qk_norm_rope")
    block_ops = {n: getattr(ops, n) for n in names}
    prewarm_block_ops(block_ops, "cpu", torch.float32, [[256]], 128, 4, 256)
    lines = [
        "prewarm block ops on cpu: 5 ops x 1 sizes",
        "tuning rms_norm_mod 1/5 (256 tokens)",
        "tuning qk_norm_rope 5/5 (256 tokens)",
        "prewarm block ops per op\n  op ",
        "prewarm block ops: done in",
    ]
    assert_phases(collect.messages, lines)
    must_fail(
        lambda: assert_phases(collect.messages, [lines[-1], lines[0]]), "order ignored"
    )


def test_startup_speed_line(collect):
    ticks = iter([0.0, 5.0, 6.0, 7.0, 8.0])
    speed = step_kernels.StartupSpeed(last=4, clock=lambda: next(ticks))
    done = [speed.on_step_end(None, None) for _ in range(4)]
    expected = "step 1 5.000 s/step, steps 2-4 median 1.000 s/step (5.0x)"
    assert expected in collect.messages, collect.messages
    assert done == [False, False, False, True]
    assert collect.messages.count(expected) == 1
