"""A run's log file: `add_log_file` after `setup` writes every project line, plain, at
the level setup set, next to the console; the next setup replaces it."""

import io
import logging

import pytest

from kohakumoss_common import log as klog
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _written(path, setup_file: bool) -> str:
    klog.setup(level=logging.INFO, stream=io.StringIO())
    if setup_file:
        klog.add_log_file(path)
    log = klog.get_logger("kohakumoss.test_log_file")
    log.info(f"step {klog.hl(3)} done", loss=0.5)
    log.debug("hidden")
    for handler in logging.getLogger("kohakumoss").handlers:
        handler.flush()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _has_the_line(text: str) -> None:
    assert "[test_log_file] [INFO] step 3 done [loss=0.5]" in text, text
    assert "\033[" not in text and "hidden" not in text


def test_the_run_log_file(tmp_path):
    _has_the_line(_written(tmp_path / "run" / "log.txt", True))
    must_fail(lambda: _has_the_line(_written(tmp_path / "x.txt", False)), "no file")
    klog.setup(level=logging.INFO, stream=io.StringIO())
    roots = [logging.getLogger(name) for name in klog.ROOTS]
    assert not any(
        isinstance(h, logging.FileHandler) for r in roots for h in r.handlers
    )
