"""Heal records: per dataset row its planned latents and caption embedding, stored as
`part<p>_<k>.pt` lists of {row, hw, latents, text} plus `empty.pt` (the empty-prompt
embedding); packed into DeviceBatches and fixed-noise model inputs."""

from pathlib import Path

import numpy as np
import torch

from kohakumoss.diffusion.objective.flow import noise_latents
from kohakumoss.types.batch import DeviceBatch
from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data.types import HostBatch

CHUNK = 512


def single_item(ds, row_id: int):
    """The row's plan item (its planned target, centre crop, no flip) and its pixels."""
    item = ds.planner.item(row_id, 0, 0, ds.planner.targets[row_id])
    return item, ds.pixels(item)


def host_batch(ds, item, pixels, prompt: str) -> HostBatch:
    """A one-sample host batch of `item` with `prompt`."""
    lf = ds.cfg.latent_factor
    h, w = item.target_hw[0] // lf, item.target_hw[1] // lf
    return HostBatch(
        ids=torch.tensor([item.row_id], dtype=torch.int64),
        hw=torch.tensor([(h, w)], dtype=torch.int32),
        text_len=torch.zeros(1, dtype=torch.int32),
        text_drop=torch.zeros(1, dtype=torch.bool),
        weight=torch.ones(1, dtype=torch.float32),
        img_keep=[None],
        pixels=[pixels],
        prompts=[prompt],
        text_limit=[0],
    )


def row_prompt(ds, row_id: int) -> str:
    """The row's caption rendered by its source's template, seeded by the row."""
    render = ds.renderers[int(ds.catalog.source[row_id])]
    return render(ds.catalog[row_id], np.random.default_rng(row_id), row_id)


def write_records(
    ds, batcher, rows: list[int], out: Path, tag: str, progress=None
) -> int:
    """Encode `rows` (DeviceBatcher with online text and latents) into
    `out/part<tag>_<k>.pt`; returns the record count."""
    chunk, idx, n = [], 0, 0
    for i, row in enumerate(rows):
        item, pixels = single_item(ds, row)
        b = batcher(host_batch(ds, item, pixels, row_prompt(ds, row)))
        chunk.append(
            {
                "row": row,
                "hw": b.seq.hw[0].tolist(),
                "latents": b.latents.to(torch.bfloat16).cpu(),
                "text": b.text.to(torch.bfloat16).cpu(),
            }
        )
        n += 1
        if len(chunk) == CHUNK or i == len(rows) - 1:
            torch.save(chunk, out / f"part{tag}_{idx:03d}.pt")
            chunk, idx = [], idx + 1
            if progress is not None:
                progress(n, len(rows))
    return n


def load_records(root: Path, n_held: int = 0, rank: int = 0, world: int = 1):
    """(held-out records: rows below n_held, rank 0 only; this rank's training records:
    rows from n_held with (row - n_held) % world == rank; the empty-prompt embedding),
    each list sorted by row."""
    held, train = [], []
    for f in sorted(Path(root).glob("part*_*.pt")):
        for r in torch.load(f, weights_only=False):
            if r["row"] < n_held:
                if rank == 0:
                    held.append(r)
            elif (r["row"] - n_held) % world == rank:
                train.append(r)
    held.sort(key=lambda r: r["row"])
    train.sort(key=lambda r: r["row"])
    return held, train, torch.load(Path(root) / "empty.pt", weights_only=False)


def make_batch(recs, drop, empty, device, rope, max_text: int = 0) -> DeviceBatch:
    """Packed batch of records (the empty prompt where `drop`; text rows capped at
    `max_text`, 0 = all)."""
    texts = [
        empty if d else (r["text"][:max_text] if max_text else r["text"])
        for r, d in zip(recs, drop)
    ]
    hw = torch.tensor([r["hw"] for r in recs], dtype=torch.int32)
    tl = torch.tensor([t.shape[0] for t in texts], dtype=torch.int32)
    seq = build_seqinfo(hw, tl, rope).to(device)
    return DeviceBatch(
        ids=torch.tensor([r["row"] for r in recs]),
        seq=seq,
        latents=torch.cat([r["latents"] for r in recs]).to(device),
        text=torch.cat(texts).to(device),
        text_drop=torch.tensor(drop, device=device),
        weight=torch.ones(len(recs), device=device),
    )


def noised_inputs(recs, empty, device, sigmas, rope, seed: int = 7) -> list[dict]:
    """Per (sigma, record) the model inputs (z, text, sigma, seq) and the flow target,
    noise drawn from `seed` per sigma (the same draws for every model)."""
    out = []
    for s in sigmas:
        gen = torch.Generator(device=device).manual_seed(seed)
        for r in recs:
            b = make_batch([r], [False], empty, device, rope)
            z0 = b.latents.float()
            eps = torch.randn(z0.shape, generator=gen, device=device)
            sigma = torch.full((1,), float(s), device=device)
            z, target = noise_latents(z0, eps, sigma, b.seq)
            out.append(
                {
                    "sigma": float(s),
                    "row": int(r["row"]),
                    "x": (z, b.text, sigma, b.seq),
                    "target": target,
                }
            )
    return out
