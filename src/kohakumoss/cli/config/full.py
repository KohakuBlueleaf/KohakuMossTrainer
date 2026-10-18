"""Full train configs: the base / layers by `use_config`, then every knob of
scripts/train/moss.py in the script's order and sections, each with the script's
one-line comment; the knobs the starter sets and the essential knobs active, the rest
commented out at their composed value."""

import ast
import io
import tokenize
from pathlib import Path

from kohakumoss.cli.config.starter import Starter, from_file, literal

ESSENTIAL = ("WEIGHTS", "DATASETS", "RUN_NAME", "STEPS")
SECTION = "# ====="


def _comments(source: str) -> dict[int, str]:
    """Line number -> comment text (without the leading `#`)."""
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.start[0]] = tok.string
    return out


def knob_entries(script: Path) -> list[tuple[str, str, str]]:
    """[("section", header, "") | ("knob", NAME, comment)] of a kogine script."""
    source = Path(script).read_text(encoding="utf-8")
    comments = _comments(source)
    out, last = [], 0
    for node in ast.parse(source).body:
        for line in range(last + 1, node.lineno):
            if comments.get(line, "").startswith(SECTION):
                out.append(("section", comments[line], ""))
        last = node.end_lineno
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        target = node.target if isinstance(node, ast.AnnAssign) else target
        if isinstance(target, ast.Name) and target.id.isupper():
            span = range(node.lineno, node.end_lineno + 1)
            text = " ".join(comments[i].lstrip("# ") for i in span if i in comments)
            out.append(("knob", target.id, text))
    return out


def _knob_lines(name: str, value, comment: str, active: bool) -> list[str]:
    lines = [f"# {comment}"] if comment else []
    body = f"{name} = {literal(value, lead=len(name) + 3)}".splitlines()
    return lines + (body if active else [f"# {line}" for line in body])


def render_full(
    st: Starter, out: Path, commands: list[str], root: Path, knobs: dict
) -> str:
    """Source of the full config of starter `st` with composed `knobs`."""
    lines = [f'"""{st.title}', "Written by kohakumoss-config; run:", ""]
    lines += [f"    {cmd}" for cmd in commands]
    lines += ['"""', "", "from kohakuengine import use_config", ""]
    lines += [
        f'use_config("{from_file(p, Path(out).parent, root)}")' for p in st.layers
    ]
    for kind, name, comment in knob_entries(root / st.script):
        if kind == "section":
            lines += ["", name]
            continue
        active = name in st.own or name in ESSENTIAL
        lines += _knob_lines(name, knobs[name], comment, active)
    return "\n".join(lines) + "\n"
