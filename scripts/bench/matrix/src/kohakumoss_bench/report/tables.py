"""Per-card kernel report from the tuning tables
(`<kernel cache>/tuning/<card>/<tag>/<kernel>.json`, rows {config, timings_ms}): for
each tuned kernel the problems decided, the configs chosen (by their `impl`
field, else the whole config), and where a table also timed a PyTorch path (`compile` /
`eager` / `torch` / `sdpa` / `w16`) the chosen config's speed against the best of those.
See docs/performance/bench-matrix.md#report.
"""

import json
import statistics
from pathlib import Path

from kohakumoss_kernels.tuning.tuner import config_name

BASELINES = ("compile", "eager", "torch", "sdpa", "w16")


def _is_baseline(key: str) -> bool:
    return any(f"={b}" in key or key.startswith(b) for b in BASELINES)


def table_report(path: Path) -> dict:
    """One kernel's table: problems, chosen counts, speedups over the PyTorch path."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    chosen: dict[str, int] = {}
    speedups = []
    for row in rows.values():
        config = row.get("config", {})
        name = config_name(config) if isinstance(config, dict) else str(config)
        label = config.get("impl") or config.get("gemm") or name
        chosen[str(label)] = chosen.get(str(label), 0) + 1
        timings = row.get("timings_ms") or {}
        base = [ms for key, ms in timings.items() if _is_baseline(key)]
        mine = timings.get(name)
        if base and mine:
            speedups.append(min(base) / mine)
    out = {"kernel": path.stem, "problems": len(rows), "chosen": chosen}
    if speedups:
        out.update(
            vs_pytorch_median=statistics.median(speedups),
            vs_pytorch_min=min(speedups),
            with_pytorch_timed=len(speedups),
        )
    return out


def card_report(tuning_root: Path) -> list[dict]:
    """Every table under one card's tuning directory (all package tags)."""
    return [
        table_report(p)
        for p in sorted(tuning_root.rglob("*.json"))
        if p.stem != "prewarm_problems"
    ]
