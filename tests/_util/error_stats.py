"""The tests' error-statistics check: a result's stats against an fp64 reference may not
be `MAX_ERROR_RATIO` x the eager reference's or more (`regressions`); every checked table
is appended to `<KOHAKUMOSS_PRECISION_DIR>/<test>.jsonl`. See
docs/internals/fp16.md#error-statistics.

A result regresses when it is non-finite where eager is finite, when its cosine deficit
is `MAX_ERROR_RATIO` x eager's or more, or when the abs AND the rel error of the same
summary (mean, or median) both are; an eager stat below the dtype's rounding scale
counts as at that scale.
"""

import json
import os
import tempfile
from pathlib import Path

from torch import Tensor

from kohakumoss_kernels.error_stats import KEYS, PAIRS, compare, worse

MAX_ERROR_RATIO = 10.0


def stats_dir() -> Path:
    """Where `record` appends tables (`KOHAKUMOSS_PRECISION_DIR`)."""
    default = os.path.join(tempfile.gettempdir(), "kohakumoss-precision")
    return Path(os.environ.get("KOHAKUMOSS_PRECISION_DIR", default))


def regressions(table: dict[str, dict], ratio: float = MAX_ERROR_RATIO) -> list[str]:
    """The stats on which the kernel is `ratio` x away from eager or more (or
    non-finite): the cosine deficit alone, abs and rel only as a pair."""
    k, e = table["kernel"], table["eager"]
    floor = table.get("floor", {})
    if not k["finite"] and e["finite"]:
        return ["non-finite"]
    away = {s for s in KEYS if k[s] > ratio * max(e[s], floor.get(s, 0.0)) and k[s] > 0}
    out = ["cos_deficit"] if "cos_deficit" in away else []
    for a, r in PAIRS:
        if a in away and r in away:
            out += [a, r]
    return out


def record(test: str, what: str, table: dict[str, dict]) -> None:
    """Append one tensor's table and worse-list to `<stats_dir>/<test>.jsonl`."""
    path = stats_dir()
    path.mkdir(parents=True, exist_ok=True)
    row = {"tensor": what, **table, "worse": worse(table)}
    with open(path / f"{test}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def assert_no_regression(
    out: Tensor, eager: Tensor, ref64: Tensor, name: str = "", test: str = "kernels"
) -> dict:
    """Record the stats table of `out` and the same-dtype `eager` result against
    `ref64`, and assert `out` has no regression; returns the table."""
    assert ref64.abs().max() > 0, f"{name}: reference is all zero"
    table = compare(out, eager, ref64)
    record(test, name, table)
    bad = regressions(table)
    assert not bad, f"{name}: regresses on {bad}: {json.dumps(table)}"
    return table
