"""Caption templates: CatalogRow + seeded generator -> caption string; options
(normaliser, dropout, shuffle, json format) are bound at construction."""

from functools import partial

import numpy as np

from kohakumoss_common.registry import build
from kohakumoss_data.caption.json_text import make_renderer
from kohakumoss_data.caption.tags import normalizer
from kohakumoss_data.registry import CAPTIONS
from kohakumoss_data.types import CatalogRow

FIELD_CHARS = {
    "tags": "tags_chars",
    "caption_nl": "nl_chars",
    "artist": "artist_chars",
    "rating": "rating_chars",
}


def _keep_values(row: CatalogRow, keep: tuple[str, ...]) -> list[str]:
    values = []
    for name in keep:
        value = getattr(row, name, None)
        match value:
            case None | "":
                continue
            case list() | tuple():
                values.extend(str(v) for v in value)
            case _:
                values.append(str(value))
    return values


def _field_chars(catalog, name: str) -> np.ndarray:
    """Char-count column of a row field (0 for a field without one)."""
    column = FIELD_CHARS.get(name)
    if column is None:
        return np.zeros(len(catalog), dtype=np.int64)
    return catalog.chars[column]


def _keep_chars(catalog, keep: tuple[str, ...], sep: int) -> np.ndarray:
    """Chars of the keep fields, each followed by a `sep`-char separator when present."""
    out = np.zeros(len(catalog), dtype=np.int64)
    for name in keep:
        chars = _field_chars(catalog, name)
        out += chars + sep * (chars > 0)
    return out


def _all(items: list, rng: np.random.Generator) -> list:
    return items


def _drop_each(p: float, items: list, rng: np.random.Generator) -> list:
    return [x for x in items if rng.random() >= p]


def _permute(items: list, rng: np.random.Generator) -> list:
    if len(items) < 2:
        return items
    return [items[i] for i in rng.permutation(len(items))]


@CAPTIONS.register("tags")
class TagTemplate:
    """Tags joined by `joiner`: keep fields first, then `keep_first` fixed tags, then
    the rest split into groups at `group_sep` tags, each group dropped with
    `group_dropout`, tags dropped with `tag_dropout`, shuffled when `shuffle`."""

    def __init__(
        self,
        keep: tuple[str, ...] = ("artist",),
        keep_first: int = 0,
        keep_sep: str = "|||",
        group_sep: str = "%%",
        shuffle: bool = True,
        tag_dropout: float = 0.0,
        group_dropout: float = 0.0,
        underscores: bool = True,
        unescape: bool = True,
        joiner: str = ", ",
        prefix: str = "",
        suffix: str = "",
    ):
        self.keep = tuple(keep)
        self.keep_first = keep_first
        self.keep_sep = keep_sep
        self.group_sep = group_sep
        self.joiner = joiner
        self.prefix = prefix
        self.suffix = suffix
        self._norm = normalizer(underscores, unescape)
        self._groups = partial(_drop_each, group_dropout) if group_dropout else _all
        self._tags = partial(_drop_each, tag_dropout) if tag_dropout else _all
        self._order = _permute if shuffle else _all

    def _split(self, tags: list[str]) -> tuple[list[str], list[list[str]]]:
        fixed, rest = tags[: self.keep_first], tags[self.keep_first :]
        if self.keep_sep in rest:
            cut = rest.index(self.keep_sep)
            fixed, rest = fixed + rest[:cut], rest[cut + 1 :]
        groups, current = [], []
        for tag in rest:
            if tag == self.group_sep:
                groups.append(current)
                current = []
            else:
                current.append(tag)
        groups.append(current)
        return fixed, groups

    def tag_list(self, row: CatalogRow, rng: np.random.Generator) -> list[str]:
        """The caption's tags in order, normalised."""
        fixed, groups = self._split(list(row.tags or []))
        kept = [self._order(self._tags(g, rng), rng) for g in self._groups(groups, rng)]
        body = [t for g in self._order(kept, rng) for t in g]
        head = _keep_values(row, self.keep) + fixed
        return [self._norm(t) for t in head + body]

    def render(self, row: CatalogRow, rng: np.random.Generator) -> str:
        return self.prefix + self.joiner.join(self.tag_list(row, rng)) + self.suffix

    def estimate_chars(self, catalog) -> np.ndarray:
        """Caption chars per row from the char columns, no dropout (an upper bound
        up to separator width)."""
        head = len(self.prefix) + len(self.suffix)
        keep = _keep_chars(catalog, self.keep, len(self.joiner))
        return head + keep + catalog.chars["tags_chars"]


@CAPTIONS.register("nl")
class NaturalTemplate:
    """Natural-language caption; keep fields are prepended as `a, b. `; falls back to
    `fallback` (a tag template) when the row has no `caption_nl`."""

    def __init__(self, keep: tuple[str, ...] = (), fallback: dict | None = None):
        self.keep = tuple(keep)
        self.fallback = build(fallback or {"template": "tags"}, CAPTIONS, "template")

    def render(self, row: CatalogRow, rng: np.random.Generator) -> str:
        if not row.caption_nl:
            return self.fallback.render(row, rng)
        head = _keep_values(row, self.keep)
        lead = (", ".join(head) + ". ") if head else ""
        return lead + row.caption_nl

    def estimate_chars(self, catalog) -> np.ndarray:
        nl = catalog.chars["nl_chars"]
        own = _keep_chars(catalog, self.keep, 2) + nl
        return np.where(nl > 0, own, self.fallback.estimate_chars(catalog))


@CAPTIONS.register("mix")
class MixTemplate:
    """Natural language with probability `p_nl` (when present), tags otherwise."""

    def __init__(
        self, p_nl: float = 0.5, nl: dict | None = None, tags: dict | None = None
    ):
        self.p_nl = p_nl
        self.nl = build(nl or {"template": "nl"}, CAPTIONS, "template")
        self.tags = build(tags or {"template": "tags"}, CAPTIONS, "template")

    def render(self, row: CatalogRow, rng: np.random.Generator) -> str:
        if row.caption_nl and rng.random() < self.p_nl:
            return self.nl.render(row, rng)
        return self.tags.render(row, rng)

    def estimate_chars(self, catalog) -> np.ndarray:
        return np.maximum(
            self.nl.estimate_chars(catalog), self.tags.estimate_chars(catalog)
        )


@CAPTIONS.register("json")
class JsonTemplate:
    """Structured caption from `row.json` plus the listed row fields."""

    def __init__(
        self,
        fields: tuple[str, ...] = ("artist", "rating", "tags", "caption_nl"),
        style: str = "lines",
        order: tuple[str, ...] = (),
        field_dropout: float = 0.0,
        keep: tuple[str, ...] = (),
    ):
        self.fields = tuple(fields)
        self._render = make_renderer(style, order, field_dropout, keep)

    def render(self, row: CatalogRow, rng: np.random.Generator) -> str:
        data = dict(row.json or {})
        for name in self.fields:
            value = getattr(row, name, None)
            if value not in (None, "", []):
                data.setdefault(name, value)
        return self._render(data, rng)

    def estimate_chars(self, catalog) -> np.ndarray:
        """The json dump plus each listed field with its `key: ` label."""
        out = catalog.chars["json_chars"].copy()
        for name in self.fields:
            chars = _field_chars(catalog, name)
            out += chars + (len(name) + 3) * (chars > 0)
        return out


def build_template(caption: dict | str | None):
    """`{"template": name, **kwargs}` -> template instance (default: tags)."""
    return build(caption or {"template": "tags"}, CAPTIONS, "template")


def _row_only(render, row: CatalogRow, rng: np.random.Generator, row_id: int) -> str:
    return render(row, rng)


def bind_template(template, source, geometry_of=None, max_chars: int = 0):
    """`render(row, rng, row_id)` of `template` over its dataset entry: the template's
    own `bind(source, geometry_of, max_chars)` when it has one (a template reading the
    source's records or the row geometry), else its `render(row, rng)`.
    `geometry_of(row_id)` gives the RowGeometry (None: the stored image, uncropped);
    `max_chars` is the caption length the text budget allows (0: unbounded)."""
    bind = getattr(template, "bind", None)
    if bind is None:
        return partial(_row_only, template.render)
    return bind(source, geometry_of, max_chars)
