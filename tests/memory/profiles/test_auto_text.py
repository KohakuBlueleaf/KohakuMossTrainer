"""`auto` counts the online text encoder with the placement each candidate plan gives
it (`mode(plan)`), from explicit text-encoder sizes, and the trainer hands that term to
the planner only when text is encoded online on the training device (model-free)."""

import pytest
import torch

from kohakumoss.memory.estimate import GIB, Workload
from kohakumoss.memory.planner import PlanReport, auto_plan
from kohakumoss.memory.profiles import resolve
from kohakumoss.offload.encoders.text_memory import TextSizes
from kohakumoss.training.build import model
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
SIZES = TextSizes(
    total=18 * GIB,
    embedding=2 * GIB,
    layers=15 * GIB,
    ring=GIB,
    hidden=4096,
    intermediate=12288,
    value_heads=32,
    head_k=128,
    head_v=128,
    elem=2,
)
WORK = Workload(PRESETS["base"], img_tokens=4096, txt_tokens=300)


def per_plan(plan) -> str:
    """A ring next to streamed DiT blocks, offload next to resident ones."""
    return "ring" if plan.stream != "off" else "offload"


def check_rows(mode) -> None:
    _, report = auto_plan(
        WORK, "cpu", budget=20 * GIB, text=(SIZES, (4096, 4096), mode), force=True
    )
    rows = {r["profile"]: r for r in report.rows if "text_memory" in r}
    assert {"speed", "balanced"} <= set(rows), rows
    assert rows["speed"]["text_memory"] == "offload", rows["speed"]
    assert rows["balanced"]["text_memory"] == "ring", rows["balanced"]
    for row in rows.values():
        # an offload pass runs after the step released its buffers: it may not raise
        # the peak; a streamed encoder keeps its resident part next to the step
        assert row["estimate_with_text"] >= row["estimate"], row
    assert rows["balanced"]["estimate_with_text"] > rows["balanced"]["estimate"]


def test_text_placement_follows_each_plan():
    check_rows(per_plan)
    must_fail(lambda: check_rows("ring"), "one fixed mode for every plan")


def _knobs(tmp_path, **over) -> dict:
    k = runnable(MEMORY_PROFILE="auto", TEXT_ENCODER=str(tmp_path), TEXT_DEVICE="")
    return {**k, **over}


def check_wiring(monkeypatch, tmp_path) -> None:
    seen = {}

    def plan_auto(work, device, overrides, **kw):
        seen["text"] = kw.get("text")
        raise model.NothingFits(PlanReport(budget=0))

    monkeypatch.setattr(model, "planned_text_sizes", lambda *a: SIZES)
    monkeypatch.setattr(model, "auto_plan", plan_auto)
    cuda = torch.device("cuda", 0)
    with pytest.raises(SystemExit):
        model.resolve_memory(_knobs(tmp_path, TEXT_CACHE="online"), cuda)
    assert seen["text"] is not None, "the online text encoder was left out"
    sizes, tokens, rule = seen["text"]
    assert sizes is SIZES and tokens[0] > 0
    assert rule(resolve("balanced", device="cpu")) == "auto"
    assert rule(resolve("speed", device="cpu")) == "auto"
    with pytest.raises(SystemExit):
        model.resolve_memory(_knobs(tmp_path, TEXT_CACHE="offline"), cuda)
    assert seen["text"] is None, "an offline text cache still counted the encoder"


def test_auto_counts_the_online_text_encoder(monkeypatch, tmp_path):
    check_wiring(monkeypatch, tmp_path)
    monkeypatch.setattr(model, "auto_text", lambda k, device: None)
    must_fail(
        lambda: check_wiring(monkeypatch, tmp_path), "text encoder left out of auto"
    )
