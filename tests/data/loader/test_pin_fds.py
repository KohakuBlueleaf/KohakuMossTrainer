"""Pinned host batches keep no worker shared-memory file open: holding a GRAD_ACC-sized
group of batches (with the trainer's SeqInfo layout) does not grow the open files."""

import os
import sys

import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data import DataConfig, Loader, MossDataset
from kohakumoss_data.types import HostBatch
from tests._util.negative import must_fail

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.budget(30),
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/self/fd"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory"),
]

HELD = 24


def attach_layout(hb):
    """Worker transform: a SeqInfo layout, as the trainer's LayoutTransform builds."""
    text_len = torch.full((len(hb.ids),), 8, dtype=torch.int32)
    hb.layout = build_seqinfo(hb.hw, text_len)
    return hb


def open_files() -> int:
    return len(os.listdir("/proc/self/fd"))


def held_growth() -> tuple[int, bool]:
    """(open files gained while holding HELD batches of a 2-worker pinned loader, the
    last batch's layout pinned)."""
    cfg = DataConfig(
        datasets=[{"source": "synthetic", "count": 64, "seed": 1}],
        target_area=128 * 128,
        max_tokens=512,
        latent_cache="off",
        text_cache="off",
    )
    loader = Loader(
        MossDataset(cfg), num_workers=2, pin_memory=True, transform=attach_layout
    )
    held, start = [], None
    for batch in loader:
        held.append(batch)
        if start is None:
            start = open_files()
        if len(held) == HELD:
            break
    return open_files() - start, held[-1].layout.cu_img.is_pinned()


def check_flat(growth: int) -> None:
    assert growth < 8, f"{growth} files opened while holding {HELD} batches"


def unpinned_layout(self):
    for name, value in self.__dict__.items():
        if isinstance(value, torch.Tensor) and not value.is_pinned():
            setattr(self, name, value.pin_memory())
    return self


def test_held_batches_keep_no_files(monkeypatch):
    growth, pinned = held_growth()
    check_flat(growth)
    assert pinned
    monkeypatch.setattr(HostBatch, "pin", unpinned_layout)
    must_fail(
        lambda: check_flat(held_growth()[0]),
        "a pin that leaves the layout in shared memory",
    )
