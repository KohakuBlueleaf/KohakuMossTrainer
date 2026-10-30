"""Metric names: a 'train' row keeps the training metrics only; wandb sees every metric
under its group, the gradient probe under probe/, the rest under detail/; LogMetrics
reads the step's device metrics in one transfer."""

import pytest
import torch

from kohakumoss.logging.keys import GROUPED, named_row, train_row
from kohakumoss.training.callbacks.monitor import LogMetrics, host_values
from kohakumoss.training.state import TrainState
from kohakumoss.types.step import StepOut
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]

ROW = {
    "loss": 0.5,
    "lr": 1e-5,
    "samples_per_s": 12.0,
    "stream_rms_q": 80.0,
    "grad_task": 3.0,
    "cache_release_time": 0.0,
    "sigma_mean": 0.6,
    "vram_peak_alloc": 150.0,
}


def _check_names(named) -> None:
    out = named(ROW)
    assert out.get("loss/total") == 0.5 and out.get("opt/lr") == 1e-5
    assert out.get("speed/img_per_s") == 12.0 and out.get("stream/rms") == 80.0
    assert out.get("probe/task") == 3.0
    assert out.get("detail/cache_release_time") == 0.0
    assert len(out) == len(ROW)


def test_named_row():
    _check_names(named_row)
    assert all("/" in v for v in GROUPED.values())


def test_named_row_negative():
    must_fail(lambda: _check_names(dict), "plain names in wandb")


def _check_train(keep) -> None:
    kept = keep(ROW)
    assert set(kept) == {"loss", "lr", "samples_per_s", "stream_rms_q", "grad_task"}


def test_train_row():
    _check_train(train_row)


def test_train_row_negative():
    must_fail(lambda: _check_train(dict), "a train row keeping every metric")


class _Sink:
    def __init__(self):
        self.rows = []

    def log(self, metrics, step):
        self.rows.append(metrics)


def test_log_metrics_levels():
    out = StepOut(torch.tensor(0.0), torch.tensor(1), {"sigma_mean": torch.tensor(0.5)})
    for level, has_sigma in (("train", False), ("bench", True)):
        sink, state = _Sink(), TrainState(step=1, loss=0.25)
        state.metrics["vram_peak_alloc"] = 1.0
        LogMetrics(1, sink, level).on_step_end(state, out)
        row = sink.rows[0]
        assert row["loss"] == 0.25
        assert ("sigma_mean" in row) is has_sigma
        assert ("vram_peak_alloc" in row) is has_sigma
        assert state.metrics == {}


def test_host_values_one_transfer(monkeypatch):
    calls = []
    real = torch.Tensor.tolist

    def counted(self):
        calls.append(self.shape)
        return real(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counted)
    got = host_values({"a": torch.tensor(1.0), "b": torch.tensor([2.0]), "c": 3})
    assert got == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert len(calls) == 1
