"""Negative-case helper: a check counts only if it fails on a broken variant."""

from collections.abc import Callable

import pytest

FAILURES = (AssertionError, pytest.fail.Exception)


def must_fail(check: Callable[[], None], what: str = "") -> None:
    """Run `check` on a deliberately broken input; it must fail, by AssertionError or
    by a pytest check (`pytest.raises` / `pytest.warns` / `pytest.fail`)."""
    try:
        check()
    except FAILURES:
        return
    raise AssertionError(f"check passed on the broken variant: {what}")
