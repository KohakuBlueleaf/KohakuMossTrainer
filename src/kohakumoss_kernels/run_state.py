"""The process-wide kernel state a run starts from: no in-memory backend failure or
status result, default backend priorities, no settled tuned choice, an open tuning
window and an empty decision log (the tuning tables and the on-disk failure record
stay)."""

from kohakumoss_kernels import registry, status
from kohakumoss_kernels.tuning import decisions, tuner, window


def reset_run_state() -> None:
    """Start a run in this process as a fresh process would (a second run, a sweep, a
    notebook)."""
    status.forget_session()
    registry.reset_priority()
    tuner.reset()
    window.reopen()
    decisions.clear()
