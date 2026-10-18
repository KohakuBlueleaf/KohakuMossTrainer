"""The tuning windows of a process: after `freeze` (the end of the prewarm) `select`
times nothing outside `tuning_window()` or `timing_here()`; after `seal` (the end of
the retune steps) kernel-vs-compile problems are no longer queued; `defer` (set at
go-live under a process group) keeps a step's missing keys untimed. See
docs/internals/tuning.md#tuning-windows."""

from contextlib import contextmanager

STATE = {"defer": False, "frozen": False, "sealed": False}


def freeze() -> None:
    """The end of the prewarm: from here `select` times nothing outside a
    `tuning_window` or `timing_here`; a missing key binds untimed (`tuner._defer`)."""
    STATE["frozen"] = True


def seal() -> None:
    """The end of the retune steps: frozen, and kernel-vs-compile problems are no
    longer queued (`sealed`)."""
    STATE["frozen"] = STATE["sealed"] = True


def frozen() -> bool:
    return STATE["frozen"]


def sealed() -> bool:
    return STATE["sealed"]


def reopen() -> None:
    """A fresh process's tuning state: not frozen, not sealed, not deferring."""
    STATE.update(defer=False, frozen=False, sealed=False)


@contextmanager
def tuning_window():
    """Inside: `select` times missing keys after `freeze` (the online text encoder's
    prewarm, a run's first sampler step); yields `close()`, which ends the window
    early. The frozen state after is the one before (frozen once sealed)."""
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
    deferral (the between-step retune, where every rank waits on the retune gate)."""
    was = STATE["defer"], STATE["frozen"]
    STATE["defer"] = STATE["frozen"] = False
    try:
        yield
    finally:
        STATE["defer"], STATE["frozen"] = was
