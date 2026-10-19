"""Script-knob checks every kogine script shares: collection from globals, unknown names
with their replacement (a rename table, dict-knob fields) or a did-you-mean hint,
choice tables, one SystemExit for every problem; `check_script` is the unknown-knob
check of a script whose knobs are the names it binds itself and its owner scripts bind,
with the owner-typed coercion of those foreign knobs. See docs/guides/configs.md."""

import ast
import difflib
import json
import os
from collections.abc import Iterable

from kohakuengine import ScriptSchema, coerce_globals, introspect_schema

from kohakumoss_common.script_schema import static_schema


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


def _py_files(folder: str) -> list[str]:
    return [
        os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(".py")
    ]


def owner_files(script: str, owners: Iterable[str]) -> list[str]:
    """`owners` as script files, in order: each path is relative to the directory of
    `script` (or absolute), and a directory stands for its .py files (sorted): a script
    family that shares one config names its own directory."""
    base = os.path.dirname(os.path.abspath(script))
    files = []
    for owner in owners:
        path = os.path.normpath(os.path.join(base, owner))
        files += _py_files(path) if os.path.isdir(path) else [path]
    return files


def owner_schema(path: str, names: set[str]) -> ScriptSchema:
    """The schema of the owner script `path` for `names`: read from its source
    (`static_schema`) when the source types or excludes every one of them, else
    `kohakuengine.introspect_schema` (imports the file; its `main()` does not run)."""
    found = static_schema(path)
    if found is None or names & found[1]:
        return introspect_schema(path)
    return found[0]


def owned_values(
    k: dict, foreign: set[str], binds: dict[str, set[str]]
) -> tuple[dict, list[str]]:
    """(coerced values, problems) of the string knobs of `k` named in `foreign`: each is
    coerced as kogine coerces a `--set` value, against the schema (`owner_schema`, once
    per owner file per call) of the first owner file of `binds` (path -> its `declared`
    names) that binds it. A name no owner binds with a default or an annotation keeps
    its value and gives a problem."""
    pending = {n: v for n, v in k.items() if n in foreign and isinstance(v, str)}
    by_owner: dict[str, dict] = {}
    for name, value in pending.items():
        owner = next((path for path, names in binds.items() if name in names), "")
        by_owner.setdefault(owner, {})[name] = value
    coerced, untyped = {}, set(by_owner.pop("", {}))
    for path, values in by_owner.items():
        schema = owner_schema(path, set(values))
        typed = {n: v for n, v in values.items() if n in schema.declared}
        coerced |= coerce_globals(
            typed, schema.defaults, annotations=schema.annotations
        )
        untyped |= set(values) - set(typed)
    problems = [
        f"{n}={pending[n]!r}: no owner script binds a default or an annotation that "
        "types it"
        for n in sorted(untyped)
    ]
    return coerced, problems


def check_script(
    g: dict,
    renamed: dict[str, str] | None = None,
    accept: Iterable[str] = (),
    owners: Iterable[str] = (),
) -> dict:
    """The knobs of a kogine script (`g` its globals after the config), after one
    SystemExit listing every knob outside the names the script, `accept` and its
    `owners` (`owner_files`) bind (with `renamed`'s replacement or a did-you-mean hint)
    and every `owned_values` problem. The string values of the knobs it takes from
    `accept` / `owners` are coerced by `owned_values` and written into `g`."""
    script = g["__file__"]
    k = collect(g)
    binds = {path: declared(path) for path in owner_files(script, owners)}
    own = declared(script)
    foreign = (set(accept) | set().union(*binds.values())) - own
    coerced, problems = owned_values(k, foreign, binds)
    exit_on(unknown_problems(k, own | foreign, renamed) + problems)
    g.update(coerced)
    return {**k, **coerced}
