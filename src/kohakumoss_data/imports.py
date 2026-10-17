"""Dotted-path resolution for user-supplied components (factories, transforms)."""

import functools
import importlib


def import_object(path: str):
    """`pkg.module:Attr.sub` (explicit module) or `pkg.module.Attr.sub` (the longest
    importable prefix is the module) -> the attribute chain."""
    if ":" in path:
        module, _, chain = path.partition(":")
        return _walk(importlib.import_module(module), chain.split("."))
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        return _walk(module, parts[cut:])
    raise ImportError(f"no importable module prefix in {path!r}")


def _walk(obj, names: list[str]):
    return functools.reduce(getattr, names, obj)
