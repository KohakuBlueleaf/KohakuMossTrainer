"""`gbc` caption template: a GBC sample (graph + metadata, `record` of its source) as one
caption. Sections come from metadata rules and the graph (`tags`, `nl`, `graph`); one of
the weighted formats orders them; each section has its own dropout, the graph its
detail-level soft drop; an over-long draw is redrawn with stronger dropout until it fits
the text budget. See docs/concepts/data.md#gbc-captions."""

import random
import re
from dataclasses import dataclass, field, replace
from functools import partial, reduce

import numpy as np

from kohakumoss_common.imports import import_object
from kohakumoss_data.catalog.text import PROMPT_CHARS
from kohakumoss_data.registry import CAPTIONS
from kohakumoss_data.types import CatalogRow, RowGeometry

try:
    from kohakugbc_data.prompt import Coords, Dag, Relation, Vertex, render
    from kohakugbc_data.prompt.detail import (
        TOP,
        apply_levels,
        first_sentence,
        sample_levels,
    )
except ImportError:
    render = None

CONTENT = ("tags", "nl", "graph")
TEXT = ("nl", "graph")
SEED_BITS = 62


def field_value(meta: dict, path: str):
    """`meta` at a dotted `path` (None when any step is missing)."""

    def step(value, name):
        return value.get(name) if isinstance(value, dict) else None

    return reduce(step, path.split("."), meta)


def items_of(value) -> list[str]:
    """A metadata value as caption items: a string, a number, or a list of them."""
    match value:
        case None | "":
            return []
        case list() | tuple():
            return [str(v) for v in value if v not in (None, "")]
        case _:
            return [str(value)]


@dataclass(frozen=True)
class MetaRule:
    """One metadata section: the value at the dotted `path` (config key `field`), then
    `bins` ([[lower, tag], ...], lower bound inclusive, highest first), `ranges`
    ([[first, last, tag], ...], both inclusive) or `map` ({value: text or [texts]});
    "" means no item. `drop`: the section's drop probability."""

    path: str
    bins: tuple = ()
    ranges: tuple = ()
    map: dict = field(default_factory=dict)
    drop: float = 0.0

    def items(self, meta: dict) -> list[str]:
        value = field_value(meta, self.path)
        if value is None:
            return []
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if self.bins:
            if not numeric:
                return []
            return items_of(next((t for lo, t in self.bins if value >= lo), ""))
        if self.ranges:
            if not numeric:
                return []
            return items_of(next((t for a, b, t in self.ranges if a <= value <= b), ""))
        if self.map:
            return items_of(self.map.get(value, ""))
        return items_of(value)


def rules_of(meta: dict) -> dict[str, MetaRule]:
    return {
        name: MetaRule(
            path=r["field"],
            bins=tuple(tuple(b) for b in r.get("bins", ())),
            ranges=tuple(tuple(x) for x in r.get("ranges", ())),
            map=dict(r.get("map", {})),
            drop=float(r.get("drop", 0.0)),
        )
        for name, r in meta.items()
    }


@dataclass(frozen=True)
class Format:
    """A weighted section order."""

    weight: float
    sections: tuple[str, ...]

    def needs(self) -> tuple[str, ...]:
        return tuple(s for s in self.sections if s in CONTENT)


@dataclass
class Parts:
    """One sample's undropped material: metadata sections, tag lists by source, the
    root's long and short captions, the graph DAG (boxes in the crop) and each vertex's
    short caption."""

    meta: dict[str, list[str]]
    tags: dict[str, list[str]]
    nl_long: str | None
    nl_short: str | None
    dag: object
    short: dict[str, str | None]
    canvas: tuple[int, int]


def crop_box(box, geo: RowGeometry):
    """A [0, 1] box of the stored image as a [0, 1] box of the crop; None when it falls
    outside or degenerates."""
    if box is None:
        return None
    left, top, right, bottom = geo.crop_ltrb
    cw, ch = right - left, bottom - top

    def x(v):
        return min(1.0, max(0.0, (v * geo.w - left) / cw))

    def y(v):
        return min(1.0, max(0.0, (v * geo.h - top) / ch))

    out = (x(box[0]), y(box[1]), x(box[2]), y(box[3]))
    return out if out[0] < out[2] and out[1] < out[3] else None


def to_dag(prompt_dag, geo: RowGeometry):
    """The Python Dag of a native PromptDag, boxes mapped into the row's crop."""
    d = prompt_dag.to_dict()
    vertices = {
        h: Vertex(
            text=v["text"],
            box=crop_box(v["box"], geo),
            tags=list(v["tags"]) if v["tags"] else None,
            children=list(v["children"]),
            relations=[Relation(list(p), t) for p, t in v["relations"]],
        )
        for h, v in d["vertices"].items()
    }
    return Dag(d["root"], vertices)


def filter_dag_tags(dag, kept):
    """`dag` with every vertex's tags passed through `kept` (None when none is left)."""
    vertices = {
        h: replace(v, tags=(kept(v.tags) or None) if v.tags else None)
        for h, v in dag.vertices.items()
    }
    return Dag(dag.root, vertices)


def drop_each(items: list[str], p: float, rng: np.random.Generator) -> list[str]:
    if p <= 0:
        return list(items)
    return [t for t in items if rng.random() >= p]


def pick(weights: list[float], rng: np.random.Generator) -> int:
    total = sum(weights)
    return int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))


@CAPTIONS.register("gbc")
class GbcTemplate:
    """Captions of a `gbc` source's samples. Arguments (docs/concepts/data.md#gbc-captions):

    `meta`: {section: rule} (MetaRule fields) or `meta_formatter`: a dotted path to
    `f(meta) -> {section: str | [str]}`; `tags`: {"sources": [{"from": "meta:<field>" |
    "graph:root", "weight"}], "drop", "shuffle", "exclude": [regex], "dedupe": [section]};
    `nl`: {"short"}; `graph`: {"keep", "tag_drop"}; `formats`: [{"weight", "sections"}];
    `list_sep` / `section_sep` / `text_sep`; `underscores`; `fit`: {"tag_drop_step",
    "tag_drop_max", "keep_step", "keep_min", "fallback", "margin"}; `prefix` /
    `suffix`."""

    geometric = True

    def __init__(
        self,
        formats: list[dict],
        meta: dict | None = None,
        meta_formatter: str = "",
        tags: dict | None = None,
        nl: dict | None = None,
        graph: dict | None = None,
        list_sep: str = ", ",
        section_sep: str = ",\n",
        text_sep: str = "\n",
        underscores: bool = True,
        fit: dict | None = None,
        prefix: str = "",
        suffix: str = "",
    ):
        if render is None:
            raise ImportError(
                "the gbc caption template needs kohakugbc-data "
                "(pip install 'kohakumoss[gbc]', Python 3.11+)"
            )
        self.formats = [
            Format(float(f["weight"]), tuple(f["sections"])) for f in formats
        ]
        if not self.formats:
            raise ValueError("gbc template: formats is empty")
        self.rules = rules_of(meta or {})
        self.meta_formatter = import_object(meta_formatter) if meta_formatter else None
        tags = dict(tags or {})
        self.tag_sources = [
            (str(s["from"]), float(s.get("weight", 1.0)))
            for s in tags.get("sources", [{"from": "graph:root"}])
        ]
        self.tag_drop = float(tags.get("drop", 0.0))
        self.shuffle = bool(tags.get("shuffle", True))
        self.exclude = [re.compile(p) for p in tags.get("exclude", ())]
        self.dedupe = tuple(tags.get("dedupe", ()))
        self.nl_short = float(dict(nl or {}).get("short", 0.0))
        graph = dict(graph or {})
        self.keep = float(graph.get("keep", 0.6))
        self.graph_tag_drop = float(graph.get("tag_drop", self.tag_drop))
        self.list_sep, self.section_sep, self.text_sep = list_sep, section_sep, text_sep
        self.underscores = underscores
        fit = dict(fit or {})
        self.tag_drop_step = float(fit.get("tag_drop_step", 0.1))
        self.tag_drop_max = float(fit.get("tag_drop_max", 0.8))
        self.keep_step = float(fit.get("keep_step", 0.1))
        self.keep_min = float(fit.get("keep_min", 0.15))
        self.margin = float(fit.get("margin", 1.0))
        self.fallback = tuple(fit.get("fallback", ()))
        self.prefix, self.suffix = prefix, suffix

    def schedule(self) -> list[tuple[float, float, float]]:
        """(tag drop, graph tag drop, keep) of each attempt, the configured one first."""
        out = [(self.tag_drop, self.graph_tag_drop, self.keep)]
        while True:
            t, g, k = out[-1]
            nxt = (
                min(self.tag_drop_max, t + self.tag_drop_step),
                min(self.tag_drop_max, g + self.tag_drop_step),
                max(self.keep_min, k - self.keep_step),
            )
            if nxt == out[-1]:
                return out
            out.append(nxt)

    def bind(self, source, geometry_of=None, max_chars: int = 0):
        if getattr(source, "record", None) is None:
            raise ValueError(
                "the gbc caption template needs a source with records (gbc)"
            )
        return BoundGbc(self, source, geometry_of, int(max_chars * self.margin))

    def render(self, row: CatalogRow, rng: np.random.Generator) -> str:
        raise ValueError("the gbc caption template renders through its source (bind)")

    def estimate_chars(self, catalog) -> np.ndarray:
        """The source's prompt bound per row."""
        return catalog.chars[PROMPT_CHARS] + len(self.prefix) + len(self.suffix)

    def _clean(self, items: list[str]) -> list[str]:
        if not self.underscores:
            return items
        return [t.replace("_", " ") for t in items]

    def parts(self, record, geo: RowGeometry) -> Parts:
        """The sample's undropped material."""
        meta = record.meta or {}
        if self.meta_formatter is not None:
            sections = {k: items_of(v) for k, v in self.meta_formatter(meta).items()}
        else:
            sections = {k: r.items(meta) for k, r in self.rules.items()}
        sections = {k: self._clean(v) for k, v in sections.items()}
        kept = partial(self._kept_tags, dupes=self._dupes(sections))
        long_dag = filter_dag_tags(to_dag(record.graph.prompt_dag(), geo), kept)
        short_dag = record.graph.prompt_dag(captions=["short"])
        short = {h: v["text"] for h, v in short_dag.to_dict()["vertices"].items()}
        root = long_dag.vertices[long_dag.root]
        tags = {}
        for name, _ in self.tag_sources:
            kind, _, key = name.partition(":")
            if kind == "graph":
                tags[name] = list(root.tags or [])
            else:
                tags[name] = kept(self._clean(items_of(field_value(meta, key))))
        nl_short = short.get(long_dag.root) or first_sentence(root.text)
        th, tw = geo.target_hw
        return Parts(sections, tags, root.text, nl_short, long_dag, short, (tw, th))

    def _dupes(self, sections: dict[str, list[str]]) -> set[str]:
        """Items of the `dedupe` sections, lower-cased."""
        return {t.lower() for name in self.dedupe for t in sections.get(name, [])}

    def _kept_tags(self, tags: list[str], dupes: set[str]) -> list[str]:
        """`tags` without the `exclude` matches and the `dedupe` sections' items."""
        return [
            t
            for t in tags
            if t.lower() not in dupes and not any(p.search(t) for p in self.exclude)
        ]

    def has(self, parts: Parts, section: str) -> bool:
        match section:
            case "tags":
                return any(parts.tags.values())
            case "nl":
                return bool(parts.nl_long)
            case "graph":
                return True
        return True

    def choose(self, parts: Parts, rng: np.random.Generator) -> Format:
        """A format whose content sections the sample has, by weight."""
        ok = [f for f in self.formats if all(self.has(parts, s) for s in f.needs())]
        if not ok:
            raise ValueError("gbc template: no format fits a sample (check `formats`)")
        return ok[pick([f.weight for f in ok], rng)]

    def tag_source(self, parts: Parts, rng: np.random.Generator) -> str:
        live = [(n, w) for n, w in self.tag_sources if parts.tags.get(n)]
        return live[pick([w for _, w in live], rng)][0] if live else ""

    def draw(self, fmt, parts: Parts, source: str, rng, attempt) -> str:
        """One caption of `fmt` under the attempt's dropout."""
        tag_drop, graph_tag_drop, keep = attempt
        out: list[tuple[str, str]] = []
        for section in fmt.sections:
            match section:
                case "tags":
                    items = drop_each(parts.tags.get(source, []), tag_drop, rng)
                    if self.shuffle and len(items) > 1:
                        items = [items[i] for i in rng.permutation(len(items))]
                    text = self.list_sep.join(items)
                case "nl":
                    use_short = parts.nl_short and rng.random() < self.nl_short
                    text = parts.nl_short if use_short else parts.nl_long or ""
                case "graph":
                    text = self.graph_text(parts, rng, keep, graph_tag_drop)
                case _:
                    rule = self.rules.get(section)
                    drop = rule.drop if rule is not None else 0.0
                    items = parts.meta.get(section, [])
                    text = (
                        ""
                        if drop and rng.random() < drop
                        else self.list_sep.join(items)
                    )
            if text:
                out.append((section, text))
        return self.prefix + self.join(out) + self.suffix

    def join(self, sections: list[tuple[str, str]]) -> str:
        """Section texts joined: `text_sep` after a text section (`nl`, `graph`), else
        `section_sep`."""
        if not sections:
            return ""
        parts = [sections[0][1]]
        for (prev, _), (_, text) in zip(sections, sections[1:]):
            parts.append(self.text_sep if prev in TEXT else self.section_sep)
            parts.append(text)
        return "".join(parts)

    def graph_text(self, parts: Parts, rng, keep: float, tag_drop: float) -> str:
        """The graph at sampled detail levels (root at full detail), each shown tag list
        dropped, rendered on the training canvas."""
        r = random.Random(int(rng.integers(1 << SEED_BITS)))
        levels = sample_levels(parts.dag, r, keep=keep, root=TOP)
        dag = apply_levels(parts.dag, levels, parts.short)
        vertices = {}
        for h, v in dag.vertices.items():
            tags = drop_each(v.tags, tag_drop, rng) if v.tags else None
            vertices[h] = replace(v, tags=tags or None)
        w, h = parts.canvas
        return render(Dag(dag.root, vertices), Coords(width=w, height=h)).rstrip("\n")


class BoundGbc:
    """`render(row, rng, row_id)` of a GbcTemplate over its source: the record and the
    row geometry read per caption, the draw redone with stronger dropout until it fits
    `max_chars` (0: unbounded), then the `fallback` sections."""

    def __init__(self, template: GbcTemplate, source, geometry_of, max_chars: int):
        self.t = template
        self.source = source
        self.geometry_of = geometry_of
        self.max_chars = max_chars
        self.attempts = template.schedule()
        self.fallback = Format(1.0, template.fallback) if template.fallback else None

    def geometry(self, row: CatalogRow, row_id: int) -> RowGeometry:
        if self.geometry_of is not None:
            return self.geometry_of(row_id)
        return RowGeometry(row.w, row.h, (row.h, row.w), (0, 0, row.w, row.h))

    def __call__(self, row: CatalogRow, rng: np.random.Generator, row_id: int) -> str:
        t = self.t
        parts = t.parts(self.source.record(row.locator), self.geometry(row, row_id))
        fmt = t.choose(parts, rng)
        source = t.tag_source(parts, rng)
        text = ""
        for attempt in self.attempts:
            text = t.draw(fmt, parts, source, rng, attempt)
            if not self.max_chars or len(text) <= self.max_chars:
                return text
        if self.fallback is not None:
            return t.draw(self.fallback, parts, source, rng, self.attempts[-1])
        return text
