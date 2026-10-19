"""Values against type annotations, for config dataclasses: int / float / bool / str,
tuples and lists (items checked), dicts, Optional and unions. See
docs/guides/configs.md#knob-types."""

import types
import typing
from dataclasses import fields

SEQUENCES = (list, tuple)


def _items(value, origin, args: tuple) -> bool:
    if not args:
        return True
    if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
        return len(value) == len(args) and all(map(matches, value, args))
    return all(matches(item, args[0]) for item in value)


def matches(value, annotation) -> bool:
    """True when `value` has the type `annotation` names: an int / float annotation
    takes an int (a float one a float too), never a bool; a tuple or list annotation
    takes a tuple or a list whose items match; a dict annotation a dict whose keys and
    values match; a union any member; any other class by isinstance; any other
    annotation form matches every value."""
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return any(matches(value, member) for member in args)
    if annotation is typing.Any:
        return True
    if annotation in (None, type(None)):
        return value is None
    if annotation in (int, float):
        allowed = (int,) if annotation is int else (int, float)
        return isinstance(value, allowed) and not isinstance(value, bool)
    if (origin or annotation) in SEQUENCES:
        return isinstance(value, SEQUENCES) and _items(value, origin, args)
    if (origin or annotation) is dict:
        if not isinstance(value, dict):
            return False
        return not args or all(
            matches(key, args[0]) and matches(item, args[1])
            for key, item in value.items()
        )
    cls = origin or annotation
    return isinstance(value, cls) if isinstance(cls, type) else True


def describe(annotation) -> str:
    """`int` -> "int"; `str | tuple[int, int, int]` -> "str | tuple[int, int, int]"."""
    if isinstance(annotation, type) and typing.get_origin(annotation) is None:
        return annotation.__name__
    return repr(annotation).replace("typing.", "")


def mismatches(obj) -> list[tuple[str, object, object]]:
    """(field name, value, annotation) of every field of the dataclass instance `obj`
    whose value does not match its annotation (`matches`)."""
    hints = typing.get_type_hints(type(obj))
    return [
        (f.name, getattr(obj, f.name), hints[f.name])
        for f in fields(obj)
        if not matches(getattr(obj, f.name), hints[f.name])
    ]
