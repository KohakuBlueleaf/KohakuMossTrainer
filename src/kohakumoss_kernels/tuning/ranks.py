"""Tuning across ranks: rank `src`'s tables to every rank (`broadcast`) and one config per
problem measured on any rank, bound on every rank (`agree`). See
docs/internals/tuning.md#multi-process.
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist

from kohakumoss_common.log import get_logger
from kohakumoss_kernels.dist import COLLECTIVE_DEVICE
from kohakumoss_kernels.hardware.current import with_index
from kohakumoss_kernels.tuning import tuner

log = get_logger(__name__)


def broadcast(src: int = 0, group=None) -> None:
    """Copy rank `src`'s tuning tables to every rank (memory and disk), then `reset`
    every rank so live calls bind from the shared tables."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    payload = [
        (
            {f"{k[0]}|{k[1]}": t.rows for k, t in tuner._TABLES.items()}
            if dist.get_rank(group) == src
            else None
        )
    ]
    dist.broadcast_object_list(payload, src=src, group=group)
    for ident, rows in payload[0].items():
        card, kernel = ident.split("|", 1)
        tuner._card_table(card, kernel).replace(rows)
    tuner._NEW.clear()
    tuner.reset()


def agreed(gathered: Sequence[Sequence[tuple]]) -> dict[tuple[str, str, str], dict]:
    """The config every rank binds per (card, kernel, key) measured since the last
    `agree`: the lowest rank's measurement (`gathered[r]`: rank r's rows)."""
    out: dict[tuple[str, str, str], dict] = {}
    for rows in gathered:
        for card, kernel, key, config in rows:
            out.setdefault((card, kernel, key), config)
    return out


def adopt(winners: dict[tuple[str, str, str], dict]) -> int:
    """Settle every winner on this process's devices of its card and store it; drops
    every binding when a winner replaces a different local choice. Returns how many
    winners differed from this process's own choice or stored row."""
    canon, chosen = tuner._canon, tuner._CHOSEN
    replaced, changed = False, 0
    rows: dict[tuple[str, str], dict[str, dict]] = {}
    for (card, kernel, key), config in winners.items():
        devices = [d for d, c in tuner._CARDS.items() if c == card]
        if not devices:
            continue
        mine = [chosen.get((d, kernel, key)) for d in devices]
        differs = any(m is not None and canon(m) != canon(config) for m in mine)
        replaced = replaced or differs
        for d in devices:
            chosen[(d, kernel, key)] = config
        row = tuner._card_table(card, kernel).get(key)
        new_row = row is None or canon(row) != canon(config)
        if new_row:
            rows.setdefault((card, kernel), {})[key] = config
        changed += new_row or differs

    for (card, kernel), configs in rows.items():
        tuner._card_table(card, kernel).put_configs(configs)
    if replaced:
        tuner._rebind()
    return changed


def positive_anywhere(count: int, group=None) -> bool:
    """Whether `count` is positive on any rank of `group` (one one-element all-reduce,
    on the backend's device)."""
    kind = COLLECTIVE_DEVICE.get(str(dist.get_backend(group)), "cpu")
    total = torch.tensor([count], device=with_index(kind))
    dist.all_reduce(total, group=group)
    return int(total.item()) > 0


def agree(group=None) -> int:
    """Between steps, on every rank: make every rank bind the same config for each
    problem some rank measured since the last call (the lowest rank's, `agreed`); a
    no-op without a process group of two or more ranks. Returns the winners that
    changed this rank's choices."""
    alone = not (dist.is_available() and dist.is_initialized())
    if alone or dist.get_world_size(group) < 2:
        tuner._NEW.clear()
        return 0
    if not positive_anywhere(len(tuner._NEW), group):
        return 0

    gathered: list = [None] * dist.get_world_size(group)
    mine = list(tuner._NEW)
    tuner._NEW.clear()
    dist.all_gather_object(gathered, mine, group=group)
    changed = adopt(agreed(gathered))
    if changed:
        log.info(f"tuning: {changed} choices agreed across ranks")
    return changed
