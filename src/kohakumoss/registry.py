"""`build(spec, REGISTRY)`: the one build-time resolution point for swappable components.

A spec is a registry name, a dotted import path, `{"name": ..., **kwargs}`, a class,
a dataclass/object with `.build()`, or an already-built instance.
"""

import importlib
from collections.abc import Callable
from typing import Any


class Registry:
    """Name -> factory map for one component family."""

    def __init__(self, family: str):
        self.family = family
        self._items: dict[str, Callable] = {}

    def register(self, name: str | None = None) -> Callable:
        """Decorator registering a class or factory under `name` (default: its name)."""

        def wrap(obj):
            self._items[name or obj.__name__] = obj
            return obj

        return wrap

    def add(self, name: str, obj: Callable) -> None:
        self._items[name] = obj

    def get(self, name: str) -> Callable:
        if name in self._items:
            return self._items[name]
        if "." in name:
            return import_object(name)
        known = ", ".join(sorted(self._items))
        raise KeyError(f"unknown {self.family} '{name}'; known: {known}")

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def import_object(path: str) -> Any:
    """Import `pkg.module.attr` (or `pkg.module:attr`)."""
    module, _, attr = path.replace(":", ".").rpartition(".")
    return getattr(importlib.import_module(module), attr)


def build(spec: Any, registry: Registry, **defaults) -> Any:
    """Resolve `spec` against `registry`; `defaults` fill kwargs the spec does not set."""
    match spec:
        case None:
            return None
        case str():
            return registry.get(spec)(**defaults)
        case dict():
            spec = dict(spec)
            factory = registry.get(spec.pop("name"))
            return factory(**{**defaults, **spec})
        case type():
            return spec(**defaults)
        case _ if hasattr(spec, "build") and callable(spec.build):
            return spec.build(**defaults)
        case _:
            return spec
