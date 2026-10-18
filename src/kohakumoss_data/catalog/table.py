"""Catalog: the rows of one or more sources as numpy planning columns plus variable
fields fetched through a per-process LRU; ids are positions, `catalog[i]` builds a
CatalogRow. See docs/concepts/data.md#the-catalog."""

from functools import lru_cache

import numpy as np

from kohakumoss_data.catalog.text import CHAR_FIELDS, text_chars
from kohakumoss_data.catalog.vault import PLANNING, CatalogVault, var_fields
from kohakumoss_data.types import CatalogRow

LRU = 4096


class RowsFetch:
    """Variable fields of in-memory rows."""

    def __init__(self, fields: list[dict]):
        self.fields = fields

    def __call__(self, i: int) -> dict:
        return self.fields[i]


class VaultFetch:
    """Variable fields from a catalog file, opened lazily per process (picklable)."""

    def __init__(self, path: str):
        self.path = path
        self._vault = None

    def __getstate__(self):
        return {"path": self.path, "_vault": None}

    def __call__(self, i: int) -> dict:
        if self._vault is None:
            self._vault = CatalogVault(self.path)
        return self._vault.fetch(i)


class ConcatFetch:
    """Variable fields of concatenated catalogs: global id -> (part, local id)."""

    def __init__(self, fetches: list, offsets: np.ndarray):
        self.fetches = fetches
        self.offsets = offsets

    def __call__(self, i: int) -> dict:
        k = int(np.searchsorted(self.offsets, i, side="right")) - 1
        return self.fetches[k](i - int(self.offsets[k]))


class Catalog:
    """`Catalog(rows)` holds in-memory rows; `Catalog.of(columns, fetch, groups)` any
    other fetch (a catalog file, a concat)."""

    def __init__(self, rows: list[CatalogRow] | None = None, source=None):
        rows = rows or []
        for i, row in enumerate(rows):
            row.id = i
        groups = list(dict.fromkeys(r.group for r in rows))
        index = {g: i for i, g in enumerate(groups)}
        chars = np.array([text_chars(r) for r in rows], dtype=np.int64).reshape(
            len(rows), len(CHAR_FIELDS)
        )
        columns = {
            "w": np.array([r.w for r in rows], dtype=np.int64),
            "h": np.array([r.h for r in rows], dtype=np.int64),
            "weight": np.array([r.weight for r in rows], dtype=np.float32),
            "has_alpha": np.array([r.has_alpha for r in rows], dtype=bool),
            "live": np.ones(len(rows), dtype=bool),
            "group_id": np.array([index[r.group] for r in rows], dtype=np.int64),
            **{name: chars[:, k] for k, name in enumerate(CHAR_FIELDS)},
        }
        self._set(columns, RowsFetch([var_fields(r) for r in rows]), groups, source)

    @classmethod
    def of(cls, columns: dict, fetch, groups: list[str], source=None) -> "Catalog":
        """A catalog over planning `columns`, a variable-field `fetch` and `groups`."""
        out = cls.__new__(cls)
        out._set(columns, fetch, groups, source)
        return out

    def _set(self, columns: dict, fetch, groups: list[str], source) -> None:
        self.w = columns["w"]
        self.h = columns["h"]
        self.weight = columns["weight"]
        self.has_alpha = columns["has_alpha"].astype(bool)
        self.live = columns["live"].astype(bool)
        self.group_id = columns["group_id"]
        self.chars = {name: columns[name] for name in CHAR_FIELDS}
        self.groups = groups
        n = len(self.w)
        self.source = np.zeros(n, dtype=np.int64) if source is None else source
        self.fetch = fetch

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_lru", None)
        return state

    def _cached(self, i: int) -> dict:
        """`fetch(i)` through this process's LRU (built on first use, never pickled)."""
        lru = self.__dict__.get("_lru")
        if lru is None:
            lru = self._lru = lru_cache(maxsize=LRU)(self.fetch)
        return lru(i)

    def columns(self) -> dict:
        """The planning columns by name (without `source`)."""
        return {
            "w": self.w,
            "h": self.h,
            "weight": self.weight,
            "has_alpha": self.has_alpha,
            "live": self.live,
            "group_id": self.group_id,
            **self.chars,
        }

    def __len__(self) -> int:
        return len(self.w)

    def __getitem__(self, i: int) -> CatalogRow:
        i = int(i)
        f = self._cached(i)
        return CatalogRow(
            id=i,
            locator=f["locator"],
            w=int(self.w[i]),
            h=int(self.h[i]),
            has_alpha=bool(self.has_alpha[i]),
            group=self.groups[int(self.group_id[i])],
            weight=float(self.weight[i]),
            tags=f["tags"],
            caption_nl=f["caption_nl"],
            artist=f["artist"],
            rating=f["rating"],
            json=f["json"],
            extra=dict(f["extra"] or {}),
        )

    def locator(self, i: int) -> str:
        """The locator of row `i`."""
        return self._cached(int(i))["locator"]

    def live_ids(self) -> np.ndarray:
        """Ids of the live (not tombstoned) rows."""
        return np.flatnonzero(self.live)

    def iter_rows(self):
        """Every live row as a CatalogRow (O(N) fetches: export / inspection only)."""
        for i in self.live_ids():
            yield self[int(i)]

    @classmethod
    def concat(cls, parts: list["Catalog"], weights: list[float]) -> "Catalog":
        """Join catalogs; part k's row weights are multiplied by `weights[k]`."""
        sizes = [len(p) for p in parts]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        names = parts[0].columns()
        columns = {n: np.concatenate([p.columns()[n] for p in parts]) for n in names}
        scaled = [p.weight * np.float32(s) for p, s in zip(parts, weights)]
        columns["weight"] = np.concatenate(scaled).astype(np.float32)
        groups, group_ids = [], []
        for p in parts:
            group_ids.append(p.group_id + len(groups))
            groups += p.groups
        columns["group_id"] = np.concatenate(group_ids)
        source = np.repeat(np.arange(len(parts), dtype=np.int64), sizes)
        fetch = ConcatFetch([p.fetch for p in parts], offsets)
        return cls.of(columns, fetch, groups, source)


def open_catalog(path: str) -> Catalog:
    """A catalog file as a lazy Catalog: the fixed columns in one read each."""
    vault = CatalogVault(path)
    columns = vault.read_fixed(names=PLANNING)
    return Catalog.of(columns, VaultFetch(path), list(vault.groups))
