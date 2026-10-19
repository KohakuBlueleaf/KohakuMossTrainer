"""Release-on-del helper: what a builder returns is freed by reference counting alone,
with the cyclic collector off. See docs/internals/reference-cycles.md."""

import gc
import weakref
from collections.abc import Callable


def _watch(build: Callable[[], tuple]) -> list[weakref.ref]:
    return [weakref.ref(o) for o in build()]


def check_released(build: Callable[[], tuple]) -> None:
    """`build()` returns the objects to watch (a root and parts it owns, e.g. a weight
    tensor); with gc disabled, each is dead once `build`'s own references are gone."""
    gc.collect()
    was = gc.isenabled()
    gc.disable()
    try:
        refs = _watch(build)
        alive = [type(r()).__qualname__ for r in refs if r() is not None]
        assert not alive, f"alive after del with gc off: {alive}"
    finally:
        if was:
            gc.enable()
        gc.collect()
