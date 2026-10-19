"""Callables that reach an object through a weak reference: the hooks and callbacks an
object installs on what it holds, without a reference cycle back to it. See
docs/internals/reference-cycles.md."""

import weakref
from collections.abc import Callable
from functools import partial


def _call(fn: Callable, ref: weakref.ref, args: tuple, *more, **kwargs):
    owner = ref()
    if owner is None:
        raise ReferenceError(
            f"{getattr(fn, '__qualname__', fn)} called after its object was freed: "
            "the hook or callback outlived the object that installed it"
        )
    return fn(owner, *args, *more, **kwargs)


def weak_call(fn: Callable, owner, *args) -> Callable:
    """`(*more, **kw) -> fn(owner, *args, *more, **kw)` holding `owner` weakly; a call
    after `owner` is gone raises ReferenceError."""
    return partial(_call, fn, weakref.ref(owner), args)


def weak_method(method) -> Callable:
    """A bound method as a `weak_call` over its function and its object."""
    return weak_call(method.__func__, method.__self__)
