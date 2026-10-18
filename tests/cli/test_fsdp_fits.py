"""The starter picks FSDP2 only when the estimated per-rank footprint is at most the
card's memory: the VRAM tolerance matches card sizes to layers, never the estimate."""

from pathlib import Path

import pytest

from kohakumoss.cli.config import presets, starter
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[2]
NEED = 30.0


def check(fits) -> None:
    """`fits(vram)` at the boundary: exactly the need fits, 3% short does not."""
    assert fits(NEED)
    assert fits(NEED * 1.5)
    assert not fits(NEED * 0.97)


def test_fsdp_fits_at_the_boundary(monkeypatch):
    monkeypatch.setattr(starter, "fsdp_need", lambda root, layers, gpus: NEED)

    def fits(vram: float) -> bool:
        c = starter.TrainChoice(weights="w", data="d", vram=vram, gpus=2, algo="full")
        return starter.fsdp_fits(ROOT, [], c)

    def tolerant(vram: float) -> bool:
        return presets.holds(vram, NEED, presets.VRAM_TOLERANCE)

    check(fits)
    must_fail(lambda: check(tolerant), "the card tolerance applied to the estimate")
    auto = starter.TrainChoice(weights="w", data="d", vram=presets.AUTO, gpus=2)
    assert not starter.fsdp_fits(ROOT, [], auto)
