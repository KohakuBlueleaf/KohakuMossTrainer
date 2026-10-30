"""METRICS_LEVEL: which metric callbacks a run attaches (train: none of the encoder /
transfer / VRAM detail; bench: VRAM peaks without the device-use breakdown; debug: all),
and the knob's choices."""

from types import SimpleNamespace

import pytest

from kohakumoss.memory.device_use import NoDeviceUse
from kohakumoss.memory.timeline import NoTimeline
from kohakumoss.training.build.loop import detail_callbacks, vram_metric_callbacks
from kohakumoss.training.callbacks.monitor import Vram, VramAtStepOne
from kohakumoss.training.knobs import CHOICES
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


class _Use:
    def metrics(self):
        return {"vram_device": 1.0}


def _check(fn) -> None:
    use, timeline = _Use(), NoTimeline()
    assert fn("train", 1, timeline, use) == []
    bench = fn("bench", 1, timeline, use)
    assert [type(c) for c in bench] == [Vram]
    assert isinstance(bench[0].device_use, NoDeviceUse)
    debug = fn("debug", 1, timeline, use)
    assert [type(c) for c in debug] == [Vram, VramAtStepOne]
    assert debug[0].device_use is use


def test_vram_metric_callbacks():
    _check(vram_metric_callbacks)


def test_vram_metric_callbacks_negative():
    def always(level, rows, timeline, use):
        return [Vram(rows, timeline, use), VramAtStepOne(use)]

    must_fail(lambda: _check(always), "VRAM metrics attached at every level")


def test_detail_callbacks_train_attaches_nothing():
    built = SimpleNamespace(engines=None, feed=object())
    text = SimpleNamespace(stats={})
    vae = SimpleNamespace(stats={})
    k = {"METRICS_LEVEL": "train"}
    assert detail_callbacks(k, built, None, 1, text, vae, "cpu") == []


def test_choices():
    assert CHOICES["METRICS_LEVEL"] == ("train", "bench", "debug")
