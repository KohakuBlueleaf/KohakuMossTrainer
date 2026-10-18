"""nvidia-smi parsing (every query field, the bus column included) and the sampler's
foreign-process record: another process holding memory on a run's card is listed in
the run's summary (the runner marks such a row `contended`, rerun on resume)."""

import pytest

pytest.importorskip(
    "kohakumoss_bench", reason="not installed: uv pip install -e scripts/bench/matrix"
)

from kohakumoss_bench.measure import smi
from kohakumoss_bench.plan.config import Card
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ROWS = [
    [
        "0",
        "GPU-a",
        "NVIDIA GeForce RTX 5090",
        "32607",
        "1200",
        "3",
        "12.0",
        "00000000:01:00.0",
    ],
    [
        "3",
        "GPU-d",
        "NVIDIA GeForce RTX 5090",
        "32607",
        "9500",
        "99",
        "12.0",
        "00000000:C1:00.0",
    ],
]


def test_card_queries_read_every_field(monkeypatch):
    monkeypatch.setattr(smi, "_query", lambda fields, what, prefix=(): ROWS)
    cards = smi.cards()
    assert [(c.index, c.bus, c.sm) for c in cards] == [
        (0, "00000000:01:00.0", 120),
        (3, "00000000:C1:00.0", 120),
    ]
    assert smi.used() == {0: 1200.0, 3: 9500.0}
    assert smi.used_mib(cards[1]) == 9500.0


def _sampled(descends) -> dict:
    sampler = smi.Sampler()
    run = sampler.register(100, Card(3, "5090", 32.0, 120, uuid="GPU-d"), 0.0)
    other = sampler.register(200, Card(0, "5090", 32.0, 120, uuid="GPU-a"), 0.0)
    smi.descends, saved = descends, smi.descends
    try:
        sampler._app_line(["GPU-d", "100", "9000"])
        sampler._app_line(["GPU-d", "101", "500"])
        sampler._app_line(["GPU-d", "777", "512"])
        sampler._app_line(["GPU-d", "888", "[N/A]"])
    finally:
        smi.descends = saved
    assert other.summary()["smi_foreign"] == {}
    return run.summary()["smi_foreign"]


def check_foreign(found: dict) -> None:
    assert set(found) == {"777"}, found


def test_foreign_process_on_the_runs_card():
    check_foreign(_sampled(lambda pid, root: pid in (root, 101)))
    must_fail(
        lambda: check_foreign(_sampled(lambda pid, root: True)), "nothing foreign"
    )
    assert not smi.descends(4242, 1, proc=smi.Path("/nonexistent"))
