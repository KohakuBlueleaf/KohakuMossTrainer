"""`kohakumoss-config` reads VRAM in exact bytes and selects a preset within the VRAM
tolerance: a 24 GB card (23.99 GiB, an RTX 4090) gets the 24g preset at the default
tolerance, the next smaller one at tolerance 0."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.cli.config import hardware, presets
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
RTX_4090 = 25_756_696_576


def _equal(a, b) -> None:
    assert a == b, (a, b)


def _detect_4090(monkeypatch) -> hardware.Accelerator:
    props = SimpleNamespace(total_memory=RTX_4090)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(hardware, "arch_of", lambda d: SimpleNamespace(has_bf16=True))
    return hardware.detect()


def test_24gb_card_selects_the_24g_preset(monkeypatch):
    acc = _detect_4090(monkeypatch)
    assert 23.98 < acc.vram_gib < 24.0, acc
    _equal(presets.vram_layer(ROOT, acc.vram_gib, False), "24g")
    _equal(presets.vram_layer(ROOT, acc.vram_gib, True), "24g_full")
    exact = presets.vram_layer(ROOT, acc.vram_gib, False, tolerance=0.0)
    must_fail(lambda: _equal(exact, "24g"), "no tolerance: one preset down")


@pytest.mark.parametrize(
    "vram, tolerance, layer",
    [(15.99, 0.05, "16g"), (15.0, 0.05, "12g"), (15.0, 0.07, "16g"), (5.9, 0.05, "6g")],
)
def test_tolerance_is_a_fraction_of_the_preset(vram, tolerance, layer):
    _equal(presets.vram_layer(ROOT, vram, False, tolerance), layer)


def test_every_card_samples_with_its_training_layer():
    _equal(presets.sample_vram_layer(ROOT, 23.99), "24g")
    _equal(presets.sample_vram_layer(ROOT, 7.9), "8g")
    _equal(presets.sample_vram_layer(ROOT, 4.0), "6g")
    _equal(presets.sample_vram_layer(ROOT, 80.0), "auto")
    must_fail(
        lambda: _equal(presets.sample_vram_layer(ROOT, 11.9, 0.0), "12g"),
        "no tolerance: one layer down",
    )
