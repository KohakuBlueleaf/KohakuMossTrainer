"""WandbLogger: an explicit mode is passed on, otherwise wandb chooses its own (WANDB_MODE,
else online; KohakuMoss passes None and never forces offline); wandb runs silent and
captures no console output, so the console matches LOGGER='console'."""

from types import SimpleNamespace

import pytest

from kohakumoss.logging import sinks
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@pytest.fixture
def fake_wandb(monkeypatch):
    seen = {}

    def init(**kw):
        seen.update(kw)
        return SimpleNamespace(log=lambda *a, **k: None, finish=lambda: None)

    monkeypatch.setattr(
        sinks, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw)
    )
    return seen


def _init(tmp_path, seen, **kw) -> dict:
    sinks.WandbLogger(str(tmp_path), "n", "p", {}, **kw).close()
    return dict(seen)


def check_defaults(seen: dict) -> None:
    assert seen["mode"] is None, "KohakuMoss picked the mode for wandb"
    assert seen["settings"] == {"silent": True, "console": "off"}, seen["settings"]


def test_wandb_default_mode_and_quiet_console(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    check_defaults(_init(tmp_path, fake_wandb))
    assert _init(tmp_path, fake_wandb, mode="offline")["mode"] == "offline"
    forced = {**_init(tmp_path, fake_wandb), "mode": "offline"}
    must_fail(lambda: check_defaults(forced), "offline forced")
