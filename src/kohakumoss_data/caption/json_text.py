"""json -> text rendering for structured captions (field order, field dropout); the
format and dropout are bound once by `make_renderer`."""

import json
from dataclasses import dataclass
from functools import partial

import numpy as np


def _value(value) -> str:
    match value:
        case str():
            return value
        case list() | tuple():
            return ", ".join(_value(v) for v in value)
        case dict():
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        case _:
            return str(value)


def _lines(data: dict, keys: list[str]) -> str:
    return "\n".join(f"{k}: {_value(data[k])}" for k in keys)


def _json(data: dict, keys: list[str]) -> str:
    return json.dumps({k: data[k] for k in keys}, ensure_ascii=False)


FORMATS = {"lines": _lines, "json": _json}


def _keep_all(keys: list[str], rng: np.random.Generator) -> list[str]:
    return keys


def _drop_fields(
    keep: tuple[str, ...], p: float, keys: list[str], rng: np.random.Generator
) -> list[str]:
    return [k for k in keys if k in keep or rng.random() >= p]


@dataclass(frozen=True)
class JsonRenderer:
    """Bound json -> text renderer: field order, field selection, format."""

    order: tuple[str, ...]
    select: object
    fmt: object

    def __call__(self, data: dict, rng: np.random.Generator) -> str:
        first = [k for k in self.order if k in data]
        keys = first + sorted(k for k in data if k not in self.order)
        return self.fmt(data, self.select(keys, rng))


def make_renderer(
    style: str = "lines",
    order: tuple[str, ...] = (),
    field_dropout: float = 0.0,
    keep: tuple[str, ...] = (),
) -> JsonRenderer:
    """`style` `lines` (`key: value` lines) or `json`; fields in `order` first, the rest
    sorted; each field not in `keep` dropped with probability `field_dropout`."""
    if style not in FORMATS:
        raise ValueError(f"json style must be one of {sorted(FORMATS)}, got {style}")
    select = (
        partial(_drop_fields, tuple(keep), field_dropout)
        if field_dropout
        else _keep_all
    )
    return JsonRenderer(tuple(order), select, FORMATS[style])
