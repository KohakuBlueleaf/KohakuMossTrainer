"""Negative-case helper: a check counts only if it fails on a broken variant."""

from collections.abc import Callable


def must_fail(check: Callable[[], None], what: str = "") -> None:
    """Run `check` on a deliberately broken input; it must raise AssertionError."""
    try:
        check()
    except AssertionError:
        return
    raise AssertionError(f"check passed on the broken variant: {what}")
