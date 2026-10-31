"""LOGGER='wandb': a resumed run continues the wandb run stored in `<output>/WANDB_ID`, a
fresh run starts a new one and stores its id; a resumed run without the file starts a
new one. Model-free: wandb.init recorded."""

from types import SimpleNamespace

import pytest

from kohakumoss.logging import sinks
from tests._util.negative import must_fail

pytestmark = [pytest.mark.cpu, pytest.mark.budget(5)]


@pytest.fixture
def inits(monkeypatch):
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id=kwargs.get("id") or f"new{len(calls)}")

    fake = SimpleNamespace(init=init, Settings=lambda **kw: None)
    monkeypatch.setattr(sinks, "wandb", fake)
    return calls


def _open(tmp_path, resume: bool) -> sinks.WandbLogger:
    logger = sinks.WandbLogger(str(tmp_path), "run", "proj", {}, resume=resume)
    logger.file.close()
    return logger


def check_resume_continues(tmp_path, inits, cls_open) -> None:
    first = cls_open(tmp_path, False)
    assert inits[-1]["id"] is None and first.run.id == "new1"
    assert (tmp_path / sinks.WANDB_ID).read_text() == "new1"
    resumed = cls_open(tmp_path, True)
    assert inits[-1]["id"] == "new1" and inits[-1]["resume"] == "allow"
    assert resumed.run.id == "new1"
    fresh = cls_open(tmp_path, False)
    assert inits[-1]["id"] is None and fresh.run.id != "new1"


def test_resume_continues_the_stored_run(tmp_path, inits):
    check_resume_continues(tmp_path, inits, _open)


def test_negative_resume_starts_a_new_run(tmp_path, inits):
    def ignores_resume(path, resume):
        return _open(path, False)

    must_fail(
        lambda: check_resume_continues(tmp_path, inits, ignores_resume),
        "resume ignored",
    )


def test_resume_without_stored_id_starts_new(tmp_path, inits):
    logger = _open(tmp_path, True)
    assert inits[-1]["id"] is None and logger.run.id == "new1"
