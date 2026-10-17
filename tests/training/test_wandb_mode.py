"""WandbLogger mode: an explicit argument, else WANDB_MODE, else offline."""

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

    monkeypatch.setattr(sinks, "wandb", SimpleNamespace(init=init))
    return seen


def _mode(tmp_path, seen, **kw) -> str:
    sinks.WandbLogger(str(tmp_path), "n", "p", {}, **kw).close()
    return seen["mode"]


def test_mode_follows_env(tmp_path, monkeypatch, fake_wandb):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert _mode(tmp_path, fake_wandb) == "disabled"
    assert _mode(tmp_path, fake_wandb, mode="online") == "online"
    monkeypatch.delenv("WANDB_MODE")
    assert _mode(tmp_path, fake_wandb) == "offline"
    must_fail(lambda: _eq(_mode(tmp_path, fake_wandb), "disabled"), "env ignored")


def _eq(a, b):
    assert a == b
