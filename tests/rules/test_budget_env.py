"""The suite's time budget (tests/conftest.py `_budget`): KOHAKUMOSS_TEST_BUDGET=0
disables every budget, a `budget(seconds)` marker included, for a first run that
compiles the kernels; otherwise the marker, then the env value, then the default."""

from types import SimpleNamespace

import pytest

from tests._util.negative import must_fail
from tests.conftest import BUDGET, _budget

pytestmark = pytest.mark.cpu


def _item(*marks: tuple[str, tuple]):
    found = [SimpleNamespace(name=name, args=args) for name, args in marks]
    by_name = {m.name: m for m in found}
    return SimpleNamespace(
        iter_markers=lambda: iter(found), get_closest_marker=by_name.get
    )


def _marker_first(item) -> float:
    mark = item.get_closest_marker("budget")
    if mark is not None:
        return float(mark.args[0])
    return _budget(item)


def check_zero_disables(budget_of) -> None:
    assert budget_of(_item(("budget", (35,)))) == 0.0
    assert budget_of(_item()) == 0.0


def test_zero_disables_every_budget(monkeypatch):
    monkeypatch.setenv("KOHAKUMOSS_TEST_BUDGET", "0")
    check_zero_disables(_budget)
    must_fail(lambda: check_zero_disables(_marker_first), "the marker wins over =0")


def test_marker_then_env_then_default(monkeypatch):
    monkeypatch.delenv("KOHAKUMOSS_TEST_BUDGET", raising=False)
    assert _budget(_item()) == BUDGET
    assert _budget(_item(("budget", (35,)))) == 35.0
    assert _budget(_item(("slow", ()))) == 0.0
    monkeypatch.setenv("KOHAKUMOSS_TEST_BUDGET", "20")
    assert _budget(_item()) == 20.0
    assert _budget(_item(("budget", (35,)))) == 35.0
