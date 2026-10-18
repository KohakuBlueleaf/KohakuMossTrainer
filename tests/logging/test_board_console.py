"""LOGGER='board' leaves the console as LOGGER='console' shows it: KohakuBoard logs to
its own file only and captures no stdout / stderr (model-free: a recording Board)."""

from types import SimpleNamespace

import pytest

from kohakumoss.logging import sinks
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@pytest.fixture
def recorded(monkeypatch):
    seen = {}

    def board(**kw):
        seen["board"] = kw
        return SimpleNamespace(finish=lambda: None)

    def settings(**kw):
        seen["log"] = kw

    monkeypatch.setattr(sinks, "Board", board)
    monkeypatch.setattr(sinks, "board_log_settings", settings)
    return seen


def check_quiet(seen: dict) -> None:
    assert seen["board"]["capture_output"] is False, seen
    assert seen["log"]["file_only"] is True and seen["log"]["log_file"].name, seen


def test_board_keeps_the_console(tmp_path, recorded):
    sinks.BoardLogger(str(tmp_path), "n", "p", {}).close()
    check_quiet(recorded)
    loud = {**recorded, "board": {**recorded["board"], "capture_output": True}}
    must_fail(lambda: check_quiet(loud), "stdout captured")
