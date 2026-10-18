"""The tuning windows of a process: after `freeze` (the end of the prewarm) `select`
times nothing outside `tuning_window()` or `timing_here()`; after `seal` (the end of
the retune steps) kernel-vs-compile problems are no longer queued; `defer` (set at
go-live under a process group) keeps every later missing key outside `timing_here()`
untimed. See docs/internals/tuning.md#tuning-windows."""

from contextlib import contextmanager

STATE = {"defer": False, "frozen": False, "sealed": False}


def freeze() -> None:
    """The end of the prewarm: from here `select` times nothing outside a
    `tuning_window` or `timing_here`; a missing key binds untimed
    (`tuner._bind_untimed`)."""
    STATE["frozen"] = True


def defer(on: bool) -> None:
    """Set the go-live deferral (`tuner.defer_if_grouped`): while on, `select` times no
    missing key outside `timing_here`."""
    STATE["defer"] = bool(on)


def seal() -> None:
    """The end of the retune steps: frozen, and kernel-vs-compile problems are no
    longer queued (`sealed`)."""
    STATE["frozen"] = STATE["sealed"] = True


def frozen() -> bool:
    return STATE["frozen"]


def sealed() -> bool:
    return STATE["sealed"]


def reopen() -> None:
    """Not frozen, not sealed, not deferring (the `tuner.set_cap` budgets are dropped
    separately, `tuner.clear_caps`; the decision log's live mode stays)."""
    STATE.update(defer=False, frozen=False, sealed=False)


@contextmanager
def tuning_window():
    """Inside: `select` times missing keys after `freeze` (the online text encoder's
    prewarm, a run's first sampler step), unless the go-live deferral is set (`defer`
    stays; `timing_here` lifts it); yields `close()`, which ends the window early. The
    frozen state after is the one before (frozen once sealed)."""
    was = STATE["frozen"]
    STATE["frozen"] = False

    def close() -> None:
        STATE["frozen"] = was or STATE["sealed"]

    try:
        yield close
    finally:
        close()


@contextmanager
def timing_here():
    """Inside: `select` times missing keys even after `freeze` and the go-live
    deferral (the between-step retune)."""
    was = STATE["defer"], STATE["frozen"]
    STATE["defer"] = STATE["frozen"] = False
    try:
        yield
    finally:
        STATE["defer"], STATE["frozen"] = was
