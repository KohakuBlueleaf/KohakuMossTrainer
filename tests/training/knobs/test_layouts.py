"""PARALLEL is a registry slot: each kind's Layout carries its launch, data, driver and
checkpoint decisions, and a dotted path names a Layout built elsewhere (model-free)."""

import dataclasses

import pytest
import torch

from kohakumoss.training import layouts
from kohakumoss.training.checks.parallel import (
    ddp_problems,
    fsdp_problems,
    parallel_problems,
)
from kohakumoss.training.context import RunContext
from kohakumoss.training.knobs import CHOICES, validate
from kohakumoss.training.layouts import LAYOUTS, Layout, layout_of
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu
CUSTOM = dataclasses.replace(LAYOUTS.get("ddp"), name="custom")
CUSTOM_PROBLEM = "the custom layout's own check"
CHECKED = dataclasses.replace(
    LAYOUTS.get("ddp"), name="checked", problems=lambda k: [CUSTOM_PROBLEM]
)
CTX = RunContext({}, None, None, torch.device("cpu"), world=2)
EXPECTED = {
    "none": (False, False, False),
    "ddp": (True, False, False),
    "fsdp": (True, False, False),
    "pipeline": (True, True, False),
    "threaded": (False, True, True),
}


def _row(layout: Layout) -> tuple:
    return layout.spawns, layout.pipelined, layout.stage_rng


def check_kinds(table: dict) -> None:
    assert set(table) == set(EXPECTED)
    for name, layout in table.items():
        assert layout.name == name
        assert _row(layout) == EXPECTED[name], name


def test_every_kind_is_registered():
    table = {name: LAYOUTS.get(name) for name in LAYOUTS.names()}
    check_kinds(table)
    assert "PARALLEL" not in CHOICES
    broken = {**table, "threaded": dataclasses.replace(table["threaded"], spawns=True)}
    must_fail(lambda: check_kinds(broken), "threaded spawning one process per device")


def test_dotted_path_names_a_layout():
    assert layout_of({"PARALLEL": f"{__name__}.CUSTOM"}) is CUSTOM
    assert layout_of({"PARALLEL": "ddp"}) is LAYOUTS.get("ddp")


class _Probe:
    """Stands in for a driver class: records the kwargs the layout builds it with."""

    def __init__(self, **kw):
        self.kw = kw


def check_drivers(table: dict, monkeypatch) -> None:
    monkeypatch.setattr(layouts, "ThreadedDriver", type("T", (_Probe,), {}))
    monkeypatch.setattr(layouts, "PipelineDriver", type("P", (_Probe,), {}))
    built = {
        name: table[name].driver(CTX, {"steps": 1}, None, "engine")
        for name in ("threaded", "pipeline")
    }
    assert type(built["threaded"]).__name__ == "T"
    assert built["threaded"].kw == {"steps": 1, "pipe": "engine"}
    assert type(built["pipeline"]).__name__ == "P"
    assert built["pipeline"].kw == {"steps": 1, "engine": "engine", "world": 2}


def check_layout_hook(problems_of) -> None:
    k = runnable(PARALLEL=f"{__name__}.CHECKED")
    assert CUSTOM_PROBLEM in problems_of(k), problems_of(k)


def test_a_dotted_layout_runs_its_own_checks():
    check_layout_hook(validate)
    kinds = {"ddp": ddp_problems, "fsdp": fsdp_problems}
    by_kind = lambda k: parallel_problems(k, kinds.get(k["PARALLEL"]))  # noqa: E731
    must_fail(lambda: check_layout_hook(by_kind), "checks branching on kind names")


def test_pipelined_kinds_pick_their_driver(monkeypatch):
    table = {name: LAYOUTS.get(name) for name in LAYOUTS.names()}
    check_drivers(table, monkeypatch)
    swapped = dataclasses.replace(table["threaded"], driver=table["pipeline"].driver)
    must_fail(
        lambda: check_drivers({**table, "threaded": swapped}, monkeypatch),
        "the threaded layout building the process-group pipeline driver",
    )
