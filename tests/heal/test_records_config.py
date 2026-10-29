"""Heal records (packing, the held-out / rank split) and the heal configuration."""

import pytest
import torch

from kohakumoss.heal.config import HealConfig, from_knobs, problems
from kohakumoss.heal.records import load_records, make_batch

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]


def _rec(row, h=4, w=6, t=5):
    return {
        "row": row,
        "hw": [h, w],
        "latents": torch.randn(h * w, 128).to(torch.bfloat16),
        "text": torch.randn(t, 32).to(torch.bfloat16),
    }


def test_make_batch():
    empty = torch.zeros(2, 32, dtype=torch.bfloat16)
    recs = [_rec(0), _rec(1, 2, 2, 9)]
    b = make_batch(recs, [False, True], empty, "cpu", 4)
    assert b.latents.shape == (28, 128)
    assert b.text.shape == (5 + 2, 32)
    assert b.ids.tolist() == [0, 1] and b.text_drop.tolist() == [False, True]
    capped = make_batch(recs, [False, False], empty, "cpu", 4, max_text=3)
    assert capped.text.shape == (6, 32)


def test_load_records_split(tmp_path):
    torch.save([_rec(r) for r in range(10)], tmp_path / "part0_000.pt")
    torch.save(torch.zeros(1, 32), tmp_path / "empty.pt")
    held, train, empty = load_records(tmp_path, 3, rank=0, world=2)
    assert [r["row"] for r in held] == [0, 1, 2]
    assert [r["row"] for r in train] == [3, 5, 7, 9]
    held, train, _ = load_records(tmp_path, 3, rank=1, world=2)
    assert held == [] and [r["row"] for r in train] == [4, 6, 8]
    assert empty.shape == (1, 32)


def test_config():
    c = from_knobs(
        {
            "TEACHER": "t.safetensors",
            "RECORDS": "r",
            "OUT": "o",
            "STEPS": 10,
            "ANNEAL_STEPS": 0,
            "IGNORED": 1,
        }
    )
    assert c.student_file == "t.safetensors" and c.steps == 10
    assert problems(c) == []
    assert HealConfig(**c.record()) == c
    assert problems(HealConfig("", "", ""))
    assert problems(HealConfig("", "", ""), paths=False) == problems(
        HealConfig("t", "r", "o")
    )
    for bad in (
        {"anneal_steps": 100},
        {"steps": 10, "anneal_steps": 20, "report": "x"},
        {"mode": "everything"},
        {"mode": "lokr", "lokr_target": "q"},
        {"lr_full": 0.0},
        {"conc_weight": -1.0},
        {"grad_acc": 0},
    ):
        assert problems(HealConfig("t", "r", "o", **bad)), bad
