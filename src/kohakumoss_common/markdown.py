"""Markdown table rendering shared by the reports: the column count comes from the
header, never from a literal."""

from collections.abc import Iterable, Sequence


def md_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    """The markdown lines of a table: the header row, the `---` rule (one cell per
    header), one line per row. A row of another length raises ValueError."""
    out = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row of {len(row)} cells under {len(headers)} headers")
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return out
