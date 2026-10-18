"""Script-knob checks every kogine script shares: collection from globals, unknown names
with their replacement (a rename table, dict-knob fields) or a did-you-mean hint,
choice tables, one SystemExit for every problem; `check_script` is the unknown-knob
check of a script whose knobs are the names it binds itself."""

import ast
import difflib
import json
import os
from collections.abc import Iterable


def collect(g: dict) -> dict:
    """UPPER_CASE globals of a script (after its config was applied)."""
    return {k: v for k, v in g.items() if k.isupper() and not k.startswith("_")}


def _hint(
    name: str,
    known: list[str],
    renamed: dict[str, str],
    nested: dict[str, tuple[str, Iterable[str]]],
) -> str:
    if name in renamed:
        return f": {renamed[name]}"
    for prefix, (knob, fields) in nested.items():
        field = name[len(prefix) :].lower()
        if name.startswith(prefix) and field in fields:
            return f": now {knob}[{field!r}]"
    close = difflib.get_close_matches(name, known, n=1)
    return f" (did you mean {close[0]}?)" if close else ""


def unknown_problems(
    k: dict,
    known: Iterable[str],
    renamed: dict[str, str] | None = None,
    nested: dict[str, tuple[str, Iterable[str]]] | None = None,
) -> list[str]:
    """One line per knob of `k` outside `known`: what replaced it when `renamed` (old
    name -> replacement text) lists it; `<knob>['<field>']` when it is `<prefix><FIELD>`
    of `nested` (prefix -> (dict knob, its field names)); else the closest known name.
    """
    known = sorted(known)
    renamed, nested = renamed or {}, nested or {}
    return [
        f"unknown knob {name}{_hint(name, known, renamed, nested)}"
        for name in sorted(set(k) - set(known))
    ]


LONG_CHOICES = 12
FULL_LISTS = {"MEMORY_PROFILE": "kohakumoss-config list --all"}


def _not_in(name: str, value, opts: tuple) -> str:
    if len(opts) <= LONG_CHOICES:
        return f"{name}={value!r} not in {opts}"
    close = difflib.get_close_matches(str(value), [str(o) for o in opts], 3, 0.0)
    where = FULL_LISTS.get(name)
    full = f"; full list: {where}" if where else ""
    return f"{name}={value!r} is not a choice (did you mean {', '.join(close)}?){full}"


def choice_problems(k: dict, choices: dict[str, tuple]) -> list[str]:
    """One line per knob of `k` whose value is not among its `choices`: the choices
    when there are at most LONG_CHOICES, else the 3 closest and where the full list is
    (FULL_LISTS)."""
    return [
        _not_in(name, k[name], tuple(opts))
        for name, opts in choices.items()
        if name in k and k[name] not in opts
    ]


def exit_on(problems: list[str]) -> None:
    """Raise one SystemExit listing every problem; nothing when there is none."""
    if problems:
        raise SystemExit("invalid configuration:\n  - " + "\n  - ".join(problems))


def prompt_list(value) -> list[str]:
    """A list-of-prompts knob value: a list / tuple as given; a string in list syntax
    (`["a, b", "c"]`, or `[a, b]` with unquoted items split at its commas) as that
    list; any other string one prompt, its commas kept."""
    if not isinstance(value, str):
        return [str(v) for v in value]
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return [value]
    for parse in (ast.literal_eval, json.loads):
        try:
            parsed = parse(text)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
    body = text[1:-1].strip()
    return [item.strip() for item in body.split(",")] if body else []


def declared(path: str) -> set[str]:
    """Every name the module-level code of the Python file `path` binds (assignments,
    imports, functions, classes)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    names = set()
    for node in tree.body:
        match node:
            case ast.Assign(targets=targets):
                names |= {t.id for t in targets if isinstance(t, ast.Name)}
            case ast.AnnAssign(target=ast.Name(id=name)) | ast.AugAssign(
                target=ast.Name(id=name)
            ):
                names.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                names |= {(a.asname or a.name).split(".")[0] for a in aliases}
            case ast.FunctionDef(name=name) | ast.ClassDef(name=name):
                names.add(name)
    return names


def siblings(path: str) -> set[str]:
    """`declared` over every .py file in the directory `path` (or the directory of the
    file `path`): a script family that shares one config."""
    path = os.path.abspath(path)
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".py"))
    return set().union(*(declared(os.path.join(folder, f)) for f in files))


def check_script(
    g: dict,
    renamed: dict[str, str] | None = None,
    accept: Iterable[str] = (),
) -> dict:
    """The knobs of a kogine script (`g` its globals after the config), after one
    SystemExit listing every knob outside the names the script binds and `accept`
    (with `renamed`'s replacement or a did-you-mean hint)."""
    k = collect(g)
    known = declared(g["__file__"]) | set(accept)
    exit_on(unknown_problems(k, known, renamed))
    return k
