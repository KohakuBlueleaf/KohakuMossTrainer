"""`build(choice, REGISTRY)`: the one build-time resolution point for swappable components.

A choice is a registry name, a dotted import path, `{key: name, **kwargs}` (key "name"
unless the caller names another), a class, an object with `.build()`, or a built instance.
"""

from collections.abc import Callable
from typing import Any

from kohakumoss_common.imports import import_object


class Registry:
    """Name -> factory map for one component family; dotted paths also resolve."""

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
        if "." in name or ":" in name:
            return import_object(name)
        known = ", ".join(sorted(self._items))
        raise KeyError(f"unknown {self.family} '{name}'; known: {known}")

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def build(choice: Any, registry: Registry, key: str = "name", **defaults) -> Any:
    """Resolve `choice` against `registry`; `defaults` fill kwargs the choice does not
    set; a dict choice names its factory under `key`."""
    match choice:
        case None:
            return None
        case str():
            return registry.get(choice)(**defaults)
        case dict():
            choice = dict(choice)
            factory = registry.get(choice.pop(key))
            return factory(**{**defaults, **choice})
        case type():
            return choice(**defaults)
        case _ if hasattr(choice, "build") and callable(choice.build):
            return choice.build(**defaults)
        case _:
            return choice
