"""`retrying`: a call repeated on transient failures, with exponential backoff, within a
time budget. See docs/concepts/training.md#transient-failures."""

import time
from collections.abc import Callable

from kohakumoss_common.log import get_logger

log = get_logger(__name__)


def _always(err: BaseException) -> bool:
    return True


def retrying(
    call: Callable,
    transient: tuple[type[BaseException], ...],
    what: str,
    deadline: float,
    attempts: int | None = None,
    backoff: float = 2.0,
    backoff_max: float = 30.0,
    is_transient: Callable[[BaseException], bool] = _always,
):
    """`call()`; an exception of `transient` types that `is_transient` accepts is
    retried after `backoff` * 2^n seconds (at most `backoff_max`) while the next attempt
    would start within `deadline` seconds of the first and fewer than `attempts` were
    made (None: no count limit). Each retry logs one warning naming `what`; the last
    failure raises OSError chained to it. Any other exception propagates at once."""
    start = time.monotonic()
    attempt = 0
    while True:
        try:
            return call()
        except transient as err:
            if not is_transient(err):
                raise
            attempt += 1
            wait = min(backoff * 2 ** min(attempt - 1, 32), backoff_max)
            spent = time.monotonic() - start
            out_of_tries = attempts is not None and attempt >= attempts
            if out_of_tries or spent + wait > deadline:
                raise OSError(
                    f"{what}: {attempt} attempts in {spent:.0f}s failed: {err}"
                ) from err
            log.warning(f"{what}: {err}; retry {attempt} in {wait:.0f}s")
            time.sleep(wait)
