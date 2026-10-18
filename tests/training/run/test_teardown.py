"""A run that raises still closes the online cache writer, and a writer that fails to
close is logged instead of replacing the run's own error (model-free)."""

import logging

import pytest

from kohakumoss.training import entry
from tests._util.logs import collecting
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


class Writer:
    def __init__(self, fails: bool = False):
        self.closed = False
        self.fails = fails

    def close(self) -> None:
        self.closed = True
        if self.fails:
            raise RuntimeError("disk full")


def check_closes(close) -> None:
    ok = Writer()
    close(ok)
    assert ok.closed
    with collecting(logging.WARNING) as logs:
        close(Writer(fails=True))
    assert any("did not close cleanly" in m for m in logs.messages), logs.messages
    close(None)


def _raising(writer) -> None:
    if writer is not None:
        writer.close()


def test_writer_closed_quietly():
    check_closes(entry.close_quietly)
    must_fail(lambda: _raises_assertion(lambda: check_closes(_raising)), "raises")


def _raises_assertion(fn) -> None:
    try:
        fn()
    except RuntimeError as err:
        raise AssertionError(str(err)) from err
