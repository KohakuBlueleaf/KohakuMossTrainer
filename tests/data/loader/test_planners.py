"""Planner invariants: geometry, batching, over-long modes, epochs, sharding, plan
purity against a growing online cache, planner kwargs."""

import collections

import numpy as np
import pytest
import torch

from kohakumoss.types.seqinfo import build_seqinfo
from kohakumoss_data import DataConfig, MossDataset
from kohakumoss_data.format.catalog import Catalog
from kohakumoss_data.registry import PLANNERS
from kohakumoss_data.resolution.base import epoch_order
from kohakumoss_data.resolution.batching import BatchLimits, greedy_tokens
from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_data.resolution.keep import keep_indices
from kohakumoss_data.types import CatalogRow
from tests._util.encode_fns import ByteCodes
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def catalog(n: int = 120, seed: int = 0, weights=None) -> Catalog:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        w = int(rng.integers(200, 3000))
        h = int(rng.integers(200, 3000))
        weight = 1.0 if weights is None else weights[i]
        rows.append(CatalogRow(id=i, locator=str(i), w=w, h=h, weight=weight))
    return Catalog(rows)


def text_len(row_id: int, variant: int) -> int:
    return 20 + (row_id * 7 + variant) % 150


GEO = Geometry(area=512 * 512, step=16, aspect_range=(0.5, 2.0))


def planner(mode="resize", cat=None, **kw):
    kw.setdefault("text_len", text_len)
    return PLANNERS.get(mode)(cat if cat is not None else catalog(), GEO, **kw)


def check_geometry(cat, batches, geo, square=False):
    for batch in batches:
        for it in batch:
            th, tw = it.target_hw
            assert th % geo.step == 0 and tw % geo.step == 0
            assert th * tw <= geo.area
            if not square:
                lo, hi = geo.aspect_range
                assert lo - 0.05 <= tw / th <= hi + 0.05
            row = cat[it.row_id]
            left, top, right, bottom = it.crop_ltrb
            assert 0 <= left < right <= row.w and 0 <= top < bottom <= row.h
            box_aspect = (right - left) / (bottom - top)
            assert abs(box_aspect - tw / th) < 0.02 * tw / th + 2 / (bottom - top)
            if geo.upscale == "never" and not square:
                assert th * tw <= row.w * row.h


def check_budget(p, batches, max_tokens):
    for batch in batches:
        assert sum(p.tokens(it) for it in batch) <= max_tokens


def check_greedy(p, batches, max_tokens):
    """Every batch fits, and the first sample of the next batch would have overflowed."""
    check_budget(p, batches, max_tokens)
    for cur, nxt in zip(batches, batches[1:]):
        used = sum(p.tokens(it) for it in cur)
        assert used + p.tokens(nxt[0]) > max_tokens, (used, p.tokens(nxt[0]))


def test_greedy_boundary():
    limits = BatchLimits(max_tokens=12)
    assert greedy_tokens(list("abc"), [5, 5, 5], limits) == [["a", "b"], ["c"]]
    assert greedy_tokens(list("ab"), [6, 6], limits) == [["a", "b"]]
    assert greedy_tokens(list("abc"), [6, 7, 1], limits) == [["a"], ["b", "c"]]

    def off_by_one(items, tokens, limits):
        out, cur, used = [], [], 0
        for item, t in zip(items, tokens):
            if cur and used + t >= limits.max_tokens:
                out.append(cur)
                cur, used = [], 0
            cur.append(item)
            used += t
        return out + [cur]

    must_fail(
        lambda: _assert_eq(off_by_one(list("ab"), [6, 6], limits), [["a", "b"]]),
        "batch closed at == budget",
    )


def _assert_eq(a, b):
    assert a == b


def test_resize_greedy_tokens():
    p = planner(max_tokens=4096)
    batches = p.epoch(0)
    check_geometry(p.catalog, batches, GEO)
    check_greedy(p, batches, 4096)
    order = [it for b in batches for it in b]
    positions = [o for o in epoch_order(p.catalog.weight, 0, 0).tolist()]
    assert [it.row_id for it in order] == positions
    ffd = planner(max_tokens=4096, batching="tokens_ffd")
    must_fail(lambda: check_greedy(ffd, ffd.epoch(0), 4096), "ffd is not greedy")


def test_resize_tokens_nearly_constant():
    geo = Geometry(area=512 * 512, upscale="to_area", aspect_range=(0.5, 2.0))
    p = PLANNERS.get("resize")(catalog(), geo, max_tokens=8192)
    img = [geo.tokens(it.target_hw) for b in p.epoch(0) for it in b]
    assert min(img) >= 0.85 * 1024 and max(img) <= 1024


def test_ffd_and_count():
    ffd = planner(max_tokens=4096, batching="tokens_ffd", window=32)
    check_budget(ffd, ffd.epoch(0), 4096)
    fill = np.mean([sum(ffd.tokens(it) for it in b) / 4096 for b in ffd.epoch(0)])
    assert fill > 0.75
    blind = planner(max_tokens=4096, batching="tokens_ffd", window=32, text_len=None)
    must_fail(lambda: check_budget(ffd, blind.epoch(0), 4096), "budget without text")
    count = planner(batching="count", batch_size=5)
    assert all(len(b) == 5 for b in count.epoch(0))
    assert count.unbatched[0] == len(count.catalog) % 5


@pytest.mark.parametrize("batching", ["count", "tokens"])
def test_bucket_uniform(batching):
    p = planner("bucket", batching=batching, batch_size=4, max_tokens=6000)
    batches = p.epoch(0)
    assert all(len({it.target_hw for it in b}) == 1 for b in batches)
    check_geometry(p.catalog, batches, GEO)
    if batching == "count":
        assert all(len(b) == 4 for b in batches)
    else:
        check_budget(p, batches, 6000)


@pytest.mark.parametrize("mode", ["square_crop", "shifted_square"])
def test_square(mode):
    p = planner(mode, batching="count", batch_size=4)
    batches = p.epoch(0)
    assert {it.target_hw for b in batches for it in b} == {(512, 512)}
    check_geometry(p.catalog, batches, GEO, square=True)
    boxes = [it.crop_ltrb for b in batches for it in b]
    centred = [
        GEO.center_box(
            p.catalog[it.row_id].w, p.catalog[it.row_id].h, it.target_hw, None
        )
        for b in batches
        for it in b
    ]
    assert (boxes == centred) == (mode == "square_crop")


def _overlong_count(p, e=0) -> int:
    over = 0
    for pos, row_id in enumerate(epoch_order(p.catalog.weight, p.seed, e).tolist()):
        it = p.item(row_id, e, pos, p.targets[row_id])
        over += GEO.tokens(it.target_hw) + text_len(row_id, it.caption_variant) > 1100
    return over


def test_overlong_drop():
    p = planner(max_tokens=1100)
    batches = p.epoch(0)
    check_budget(p, batches, 1100)
    expected = _overlong_count(p)
    assert expected > 0 and p.drops[0] == expected
    assert (
        sum(len(b) for b in batches)
        == len(epoch_order(p.catalog.weight, 0, 0)) - expected
    )


def test_overlong_truncate_text():
    p = planner(max_tokens=1100, overlong="truncate_text")
    batches = p.epoch(0)
    check_budget(p, batches, 1100)
    fitted = [it for b in batches for it in b if it.text_limit]
    assert len(fitted) == _overlong_count(p) > 0 and p.drops[0] == 0
    for it in fitted:
        img, text = p.sizes(it)
        assert img + text == 1100 and it.text_limit == 1100 - GEO.tokens(it.target_hw)


def test_overlong_drop_tokens_positions():
    p = planner(max_tokens=1100, overlong="drop_tokens")
    batches = p.epoch(0)
    check_budget(p, batches, 1100)
    fitted = [it for b in batches for it in b if it.img_keep]
    assert len(fitted) == _overlong_count(p) > 0
    again = planner(max_tokens=1100, overlong="drop_tokens")
    assert again.epoch(0) == batches
    for it in fitted[:20]:
        h, w = it.target_hw[0] // 16, it.target_hw[1] // 16
        keep = keep_indices(it, h * w)
        assert torch.equal(keep, keep_indices(it, h * w))
        assert (
            keep.numel()
            == it.img_keep
            == 1100 - text_len(it.row_id, it.caption_variant)
        )
        assert torch.all(keep[1:] > keep[:-1]) and keep[-1] < h * w
        hw = torch.tensor([[h, w]])
        tl = torch.tensor([5])
        full = build_seqinfo(hw, tl)
        kept = build_seqinfo(hw, tl, img_keep=[keep])
        img_pos = kept.pos_joint[kept.img_in_joint]
        assert torch.equal(img_pos, full.pos_joint[full.img_in_joint][keep])
        assert kept.n_img == keep.numel()
        wrong = keep.flip(0)
        must_fail(
            lambda: _assert_tensor(img_pos, full.pos_joint[full.img_in_joint][wrong]),
            "unsorted keep",
        )


def _assert_tensor(a, b):
    assert torch.equal(a, b)


def test_epoch_weights_and_sharding():
    weights = [1.0] * 40 + [3.0] * 10 + [0.5] * 20
    cat = catalog(70, weights=weights)
    counts = collections.Counter(epoch_order(cat.weight, 3, 0).tolist())
    assert all(counts[i] == 1 for i in range(40))
    assert all(counts[i] == 3 for i in range(40, 50))
    assert all(counts[i] <= 1 for i in range(50, 70))
    p = planner(cat=cat, max_tokens=4096, seed=3)
    n = p.epoch_length(0)
    world = 3
    seen = [
        tuple(p.plan(index, rank, world).items)
        for rank in range(world)
        for index in range(n // world)
    ]
    assert len(set(seen)) == len(seen) == (n // world) * world
    again = planner(cat=cat, max_tokens=4096, seed=3)
    assert again.plan(5, 1, world).items == p.plan(5, 1, world).items
    other = planner(cat=cat, max_tokens=4096, seed=4)
    assert other.plan(5, 1, world).items != p.plan(5, 1, world).items


def test_epochs_continue():
    p = planner(
        "bucket", cat=catalog(30), batching="count", batch_size=2, drop_last=False
    )
    n0 = p.epoch_length(0)
    assert p.plan(n0, 0, 1).items == p.epoch(1)[0]
    assert p.epoch(1) != p.epoch(0)


def test_upscale_modes():
    small = Catalog([CatalogRow(id=0, locator="0", w=300, h=200)])
    never = Geometry(area=1024 * 1024, upscale="never")
    up = Geometry(area=1024 * 1024, upscale="to_area")
    assert never.fit(300, 200) == (192, 288)
    th, tw = up.fit(300, 200)
    assert th * tw > 0.9 * 1024 * 1024 and th * tw <= 1024 * 1024
    p = PLANNERS.get("bucket")(small, never, batching="count", batch_size=1)
    assert p.epoch(0)[0][0].target_hw == (192, 288)


def data_cfg(tmp_path, **kw) -> DataConfig:
    base = dict(
        datasets=[{"source": "synthetic", "count": 40, "seed": 5}],
        resolution_mode="resize",
        batching="tokens",
        target_area=256 * 256,
        max_tokens=700,
        latent_cache="off",
        text_cache="online",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
    )
    base.update(kw)
    return DataConfig(**base)


def plan_of(ds) -> list:
    return [
        [it.row_id for it in ds.planner.plan(i, 0, 1).items]
        for i in range(ds.planner.epoch_length(0))
    ]


def test_online_text_plan_is_cache_independent(tmp_path):
    before = plan_of(MossDataset(data_cfg(tmp_path)))
    ds = MossDataset(data_cfg(tmp_path))
    encode = ByteCodes(dim=8)
    for row_id in range(len(ds.catalog)):
        caption = ds.caption(row_id, 0)
        ds.texts.put(ds.texts.key(caption), encode([caption])[0])
    _assert_eq(plan_of(MossDataset(data_cfg(tmp_path))), before)
    rendered = data_cfg(tmp_path, token_budget="render")
    _assert_eq(plan_of(MossDataset(rendered)), plan_of(MossDataset(rendered)))
    live = MossDataset(data_cfg(tmp_path, text_cache="offline", token_budget="render"))
    must_fail(lambda: _assert_eq(plan_of(live), before), "plan from live lengths")


def test_planner_kwargs_override(tmp_path):
    kwargs = {"max_samples": 2, "window": 8}
    ds = MossDataset(data_cfg(tmp_path, text_cache="off", planner_kwargs=kwargs))
    sizes = [
        len(ds.planner.plan(i, 0, 1).items) for i in range(ds.planner.epoch_length(0))
    ]
    assert max(sizes) == 2
