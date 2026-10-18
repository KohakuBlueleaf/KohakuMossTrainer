"""KohakuMoss logging for the `kohakumoss` package: `kohakumoss_common.log` (get_logger,
setup, set_level, hl, once-only helpers) plus tqdm progress bars that follow the level
and a timed phase helper. See docs/guides/logging.md."""

import logging
import os
import sys
import time
from contextlib import contextmanager

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from kohakumoss_common.log import (
    ROOTS,
    Formatter,
    KMossLogger,
    add_log_file,
    get_logger,
    hl,
    info_once,
    log_object,
    log_table,
    render,
    set_level,
    setup,
    supports_colour,
    table,
    warning_once,
)

__all__ = [
    "Formatter",
    "KMossLogger",
    "add_log_file",
    "bar",
    "bars",
    "get_logger",
    "hl",
    "info_once",
    "interactive",
    "log_object",
    "log_table",
    "render",
    "table",
    "progress_enabled",
    "set_level",
    "setup",
    "supports_colour",
    "timed",
    "warning_once",
]


def interactive(stream=None) -> bool:
    """Whether a live bar suits `stream` (default stderr): a TTY, or FORCE_COLOR set."""
    stream = sys.stderr if stream is None else stream
    if os.environ.get("FORCE_COLOR", "") not in ("", "0"):
        return True
    return bool(getattr(stream, "isatty", None) and stream.isatty())


def progress_enabled() -> bool:
    """Bars are shown while the project logs at INFO or lower."""
    return logging.getLogger(ROOTS[0]).isEnabledFor(logging.INFO)


def bar(iterable=None, total: int | None = None, desc: str = "", enabled=True, **kw):
    """A tqdm bar on stderr; disabled when `enabled` is False or above INFO."""
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        file=sys.stderr,
        disable=not (enabled and progress_enabled()),
        dynamic_ncols=True,
        **kw,
    )


@contextmanager
def bars():
    """Project log lines go through tqdm.write while bars are open."""
    with logging_redirect_tqdm(loggers=[logging.getLogger(name) for name in ROOTS]):
        yield


@contextmanager
def timed(log: logging.Logger, what: str):
    """Log `what ...` on entry and `what: done` with `secs` and the fields the body put
    into the yielded dict on exit (nothing on an exception)."""
    fields: dict = {}
    log.info(f"{what} ...", stacklevel=3)
    start = time.perf_counter()
    yield fields
    log.info(
        f"{what}: done",
        secs=round(time.perf_counter() - start, 2),
        stacklevel=3,
        **fields,
    )
