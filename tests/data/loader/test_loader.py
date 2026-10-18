"""Stateless loader: identical stream for any worker count, exact resume, packing, the
in-worker transform."""

import hashlib
import os

import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data import DataConfig, Loader, MossDataset
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def cfg(**kw) -> DataConfig:
    base = dict(
        datasets=[{"source": "synthetic", "count": 32, "seed": 1}],
        resolution_mode="resize",
        target_area=256 * 256,
        max_tokens=1024,
        latent_cache="off",
        text_cache="off",
        seed=7,
        text_drop=0.2,
    )
    base.update(kw)
    return DataConfig(**base)


def fingerprint(batch) -> tuple:
    pixels = hashlib.blake2b(b"".join(p.tobytes() for p in batch.pixels)).hexdigest()
    return (
        batch.index,
        tuple(batch.ids.tolist()),
        tuple(batch.hw.flatten().tolist()),
        tuple(batch.prompts),
        pixels,
    )


def take(loader, n) -> list:
    out = []
    for batch in loader:
        out.append(fingerprint(batch))
        if len(out) == n:
            break
    return out


def check_same(a, b) -> None:
    assert a == b


def test_worker_count_invariance():
    ds = MossDataset(cfg())
    single = take(Loader(ds, num_workers=0, pin_memory=False), 6)
    multi = take(Loader(ds, num_workers=2, pin_memory=False), 6)
    check_same(single, multi)
    assert [f[0] for f in single] == list(range(6))
    other = take(Loader(MossDataset(cfg(seed=8)), num_workers=0, pin_memory=False), 6)
    must_fail(lambda: check_same(single, other), "different seed")


def test_resume_exact():
    ds = MossDataset(cfg())
    full = take(Loader(ds, pin_memory=False), 8)
    first = Loader(ds, pin_memory=False)
    head = take(first, 3)
    state = first.state_dict()
    assert state == {"seed": 7, "start": 3}
    resumed = Loader(MossDataset(cfg()), pin_memory=False)
    resumed.load_state_dict(state)
    assert head + take(resumed, 5) == full
    with pytest.raises(ValueError):
        Loader(MossDataset(cfg(seed=9)), pin_memory=False).load_state_dict(state)


def test_ranks_disjoint():
    ds = MossDataset(cfg())
    r0 = [ds.materialize(i, 0, 2).ids.tolist() for i in range(4)]
    r1 = [ds.materialize(i, 1, 2).ids.tolist() for i in range(4)]
    plans = [tuple(ds.planner.plan(i, r, 2).items) for r in range(2) for i in range(4)]
    assert len(set(plans)) == 8
    assert r0 != r1


def test_packed_layout_matches_seqinfo():
    ds = MossDataset(cfg(text_cache="off"))
    b = ds.materialize(2)
    n_img = sum(p.shape[0] // 16 * (p.shape[1] // 16) for p in b.pixels)
    seq = build_seqinfo(b.hw, torch.ones(len(b.pixels), dtype=torch.int32))
    assert seq.n_img == n_img
    assert [tuple(p.shape[:2]) for p in b.pixels] == [
        (h * 16, w * 16) for h, w in b.hw.tolist()
    ]


def tag_pid(batch):
    return batch, os.getpid()


def test_transform_runs_in_worker():
    ds = MossDataset(cfg())
    out = []
    for batch, pid in Loader(ds, num_workers=2, pin_memory=False, transform=tag_pid):
        out.append((batch.index, batch.ids.tolist(), pid))
        if len(out) == 4:
            break
    assert [o[0] for o in out] == [0, 1, 2, 3]
    assert all(pid != os.getpid() for *_, pid in out)
    named = Loader(
        ds, pin_memory=False, transform="tests.data.loader.test_loader.tag_pid"
    )
    first, pid = next(iter(named))
    assert first.ids.tolist() == out[0][1] and pid == os.getpid()
