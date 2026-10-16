"""Name -> factory registries for the data package (sources, planners, captions)."""

import importlib
from collections.abc import Callable
from typing import Any


class Registry:
    """Name -> factory map for one component family; dotted paths also resolve."""

    def __init__(self, family: str):
        self.family = family
        self._items: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        def wrap(obj):
            self._items[name] = obj
            return obj

        return wrap

    def get(self, name: str) -> Callable:
        if name in self._items:
            return self._items[name]
        if "." in name:
            module, _, attr = name.replace(":", ".").rpartition(".")
            return getattr(importlib.import_module(module), attr)
        known = ", ".join(sorted(self._items))
        raise KeyError(f"unknown {self.family} '{name}'; known: {known}")

    def names(self) -> list[str]:
        return sorted(self._items)


def build(spec: Any, registry: Registry, key: str = "name", **defaults) -> Any:
    """Resolve a name / {key: name, **kwargs} / class / instance against `registry`."""
    match spec:
        case None:
            return None
        case str():
            return registry.get(spec)(**defaults)
        case dict():
            spec = dict(spec)
            factory = registry.get(spec.pop(key))
            return factory(**{**defaults, **spec})
        case type():
            return spec(**defaults)
        case _:
            return spec


SOURCES = Registry("source")
PLANNERS = Registry("planner")
CAPTIONS = Registry("caption template")
