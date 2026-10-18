"""Log capture for tests: a handler that keeps records, attached to every package root,
and an in-order phase check over the captured messages."""

import logging
from contextlib import contextmanager

from kohakumoss_common import log as klog


class Collect(logging.Handler):
    """Keeps every record it receives."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [r.getMessage() for r in self.records]


@contextmanager
def collecting(level: int = logging.INFO):
    """A Collect on the three package roots at `level`; removed on exit."""
    handler = Collect()
    roots = [logging.getLogger(name) for name in klog.ROOTS]
    for root in roots:
        root.addHandler(handler)
        root.setLevel(level)
    try:
        yield handler
    finally:
        for root in roots:
            root.removeHandler(handler)


def assert_phases(messages: list[str], phases: list[str]) -> None:
    """Each phase prefix appears in `messages`, in this order."""
    at = 0
    for phase in phases:
        hits = [i for i, m in enumerate(messages) if m.startswith(phase) and i >= at]
        assert hits, f"{phase!r} missing after #{at}: {messages}"
        at = hits[0] + 1
