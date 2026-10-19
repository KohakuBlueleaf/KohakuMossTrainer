"""The knob schema of a kogine script read from its source, without importing it: the
defaults and annotations that evaluate from literals, arithmetic, builtin types and
earlier such knobs, and the names bound any other way. See
docs/guides/configs.md#knob-types."""

import ast
import operator

from kohakuengine import ScriptSchema

BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitOr: operator.or_,
}
UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
TYPES = {
    t.__name__: t
    for t in (bool, bytes, dict, float, frozenset, int, list, set, str, tuple)
}
FAILURES = (ArithmeticError, LookupError, TypeError, ValueError)


class Opaque(Exception):
    """An expression the source alone does not evaluate."""


def evaluate(node: ast.expr, names: dict):
    """The value of the expression `node`: constants, tuple / list / set / dict
    displays, unary and binary arithmetic (`|` included), subscripts and the names of
    `names`; Opaque for any other form or a failing operation."""
    try:
        match node:
            case ast.Constant(value=value):
                return value
            case ast.Name(id=name) if name in names:
                return names[name]
            case ast.Tuple(elts=items):
                return tuple(evaluate(item, names) for item in items)
            case ast.List(elts=items):
                return [evaluate(item, names) for item in items]
            case ast.Set(elts=items):
                return {evaluate(item, names) for item in items}
            case ast.Dict(keys=keys, values=values) if None not in keys:
                pairs = zip(keys, values)
                return {evaluate(k, names): evaluate(v, names) for k, v in pairs}
            case ast.UnaryOp(op=op, operand=operand) if type(op) in UNARY:
                return UNARY[type(op)](evaluate(operand, names))
            case ast.BinOp(left=left, op=op, right=right) if type(op) in BINARY:
                return BINARY[type(op)](evaluate(left, names), evaluate(right, names))
            case ast.Subscript(value=value, slice=index):
                return evaluate(value, names)[evaluate(index, names)]
    except FAILURES as exc:
        raise Opaque(ast.dump(node)) from exc
    raise Opaque(ast.dump(node))


def _main_guard(node: ast.stmt) -> bool:
    match node:
        case ast.If(
            test=ast.Compare(
                left=ast.Name(id="__name__"),
                ops=[ast.Eq()],
                comparators=[ast.Constant(value="__main__")],
            )
        ):
            return True
    return False


def _value(node: ast.expr, values: dict):
    try:
        return evaluate(node, values)
    except Opaque:
        return Opaque


def _annotation(node: ast.expr):
    found = _value(node, TYPES)
    if not isinstance(found, str):
        return found
    try:
        return _value(ast.parse(found, mode="eval").body, TYPES)
    except SyntaxError:
        return Opaque


def static_schema(path: str) -> tuple[ScriptSchema, set[str]] | None:
    """(schema of the knobs typed from the source, names bound any other way) of the
    module-level code of the Python file `path`, as `kohakuengine.introspect_schema`
    reads it for those names; None when that code runs any statement other than
    imports, definitions, plain-name assignments, constant expressions and the
    `__main__` guard."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    values, annotations, opaque, untyped = {}, {}, set(), set()
    for node in tree.body:
        targets, value = [], None
        match node:
            case ast.Assign(targets=found, value=value) if all(
                isinstance(t, ast.Name) for t in found
            ):
                targets = [t.id for t in found]
            case ast.AnnAssign(target=ast.Name(id=name), annotation=hint, value=value):
                targets = [name] if value is not None else []
                annotations[name] = _annotation(hint)
                if annotations[name] is Opaque:
                    untyped.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                targets = [(a.asname or a.name).split(".")[0] for a in aliases]
            case (
                ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                targets = [name]
            case ast.Expr(value=ast.Constant()):
                pass
            case _ if _main_guard(node):
                pass
            case _:
                return None
        result = Opaque if value is None else _value(value, values)
        for name in targets:
            values.pop(name, None)
            opaque.discard(name)
            if result is Opaque:
                opaque.add(name)
            else:
                values[name] = result
    for name in untyped:
        annotations.pop(name)
    return ScriptSchema(values, annotations), opaque | untyped
