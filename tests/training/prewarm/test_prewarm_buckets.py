"""Kernel prewarm planning: planned length buckets (one replay point per bucket, none
above the planned maximum), the attention calls within it, and the prewarmed devices."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kohakumoss.training.build.kernels import adapter_prewarm_devices
from kohakumoss.tuning.lengths import (
    RECORD_JOINT,
    RECORD_TXT,
    block_lengths,
    image_tokens_max,
    planned_lengths,
    planned_segments,
    token_maps,
)
from kohakumoss.tuning.prewarm import ModelPrewarm, prewarm_devices
from kohakumoss_data import DataConfig
from kohakumoss_data.dataset import MossDataset
from kohakumoss_kernels.recording import Call, Snap
from kohakumoss_kernels.tuning.buckets import mean_bucket, pow2, segment_bucket
from kohakumoss_kernels.tuning.prewarm import (
    attention_lengths,
    pack_sizes,
    prewarm_attention,
    segment_counts,
)
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _side_up(square: float, step: int) -> int:
    """Smallest multiple of `step` whose square reaches `square` (integer search)."""
    w = step
    while w * w < square:
        w += step
    return w


def _ref_tokens(area: int, aspects, step: int, factor: int) -> int:
    best = 0
    for a in aspects:
        w, h = _side_up(area * a, step), _side_up(area / a, step)
        best = max(best, (w // factor) * (h // factor))
    return best


def _floored_tokens(cfg) -> int:
    best = 0
    for a in (*cfg.aspect_range, 1.0):
        w = int(math.sqrt(cfg.target_area * a)) // cfg.size_step * cfg.size_step
        h = int(math.sqrt(cfg.target_area / a)) // cfg.size_step * cfg.size_step
        best = max(best, (w // cfg.latent_factor) * (h // cfg.latent_factor))
    return best


@pytest.mark.parametrize("area", [1024 * 1024, 1000 * 1000, 768 * 768, 256 * 256])
def test_image_tokens_max(area):
    cfg = DataConfig(target_area=area, aspect_range=(0.5, 2.0))
    want = _ref_tokens(area, (0.5, 2.0, 1.0), cfg.size_step, cfg.latent_factor)
    assert image_tokens_max(cfg) == want

    def floored():
        assert _floored_tokens(cfg) == want

    must_fail(floored, "sides rounded down to the size step")


def test_planned_lengths():
    cfg = DataConfig(
        target_area=1024 * 1024, aspect_range=(0.25, 4.0), max_text_tokens=512
    )
    assert planned_lengths(cfg) == (4096 + 512, cfg.max_tokens)
    count = DataConfig(
        target_area=1024 * 1024, batching="count", batch_size=3, max_text_tokens=512
    )
    assert planned_lengths(count) == (4608, 3 * 4608)
    capped = DataConfig(target_area=4096 * 4096, max_tokens=8192)
    assert planned_lengths(capped)[0] == 8192


def _assert_covers(lengths, batch_max) -> None:
    """One replay point per token bucket up to `batch_max`'s, none above `batch_max`."""
    sizes = [sum(x) for x in lengths]
    assert max(sizes) <= batch_max, sizes
    keys = [pow2(n, 256) for n in sizes]
    assert keys == [256 << i for i in range(len(keys))], keys
    assert keys[-1] == pow2(batch_max, 256), keys


def _bucket_tops(batch_max: int) -> list[list[int]]:
    """The broken variant: every bucket at its top."""
    return [[pow2(n, 256)] for (n,) in block_lengths(batch_max)]


@pytest.mark.parametrize("batch_max", [100, 1000, 9000, 16384, 20000])
def test_block_lengths_cover_every_bucket_within_the_plan(batch_max):
    lengths = block_lengths(batch_max)
    _assert_covers(lengths, batch_max)
    assert sum(lengths[-1]) == min(batch_max, pow2(batch_max, 256))
    if len(lengths) > 1:
        must_fail(lambda: _assert_covers(lengths[:-1], batch_max), "top bucket missing")
    if pow2(batch_max, 256) != batch_max:
        must_fail(
            lambda: _assert_covers(_bucket_tops(batch_max), batch_max), "bucket tops"
        )


def test_token_maps_stay_within_the_plan():
    for joint in (m[max(m)] for m in token_maps(20000)):
        assert joint <= 20000
    assert max(m[max(m)] for m in token_maps(20000)) == 20000


def _attention_calls(prewarm, sample, segments, max_tokens, shortest) -> list:
    """(tokens, longest segment, segments) of every attention prewarm call."""
    seen = []

    def spy(q, k, v, cu, length, scale):
        seen.append((q.shape[0], length, cu.shape[0] - 1))
        return q

    prewarm(
        spy, "cpu", torch.float32, 1, 8, sample, segments=segments,
        backward=False, max_tokens=max_tokens, shortest=shortest,
    )  # fmt: skip
    return seen


def _reachable(length, segments, max_tokens, shortest) -> set[int]:
    """Segment-count buckets of every pack a batch can hold: one `length` sample plus
    shorter ones of `shortest` tokens or more, at most `segments`, `max_tokens` total.
    """
    short = min(shortest, length)
    return {
        segment_bucket(s)
        for s in range(1, segments + 1)
        if length + (s - 1) * short <= max_tokens
    }


class _Captions:
    """A dataset whose shortest caption estimate is `tokens` and smallest planned grid
    `image` tokens (MossDataset's API; 0: the area target's grid)."""

    def __init__(self, tokens: int, image: int = 0):
        self.tokens = tokens
        self.image = image

    def text_tokens_min(self) -> int:
        return self.tokens

    def image_tokens_min(self) -> int:
        return self.image


def check_attention_buckets(
    prewarm, cfg, dataset=None, planned_text=None, planned_image=None
) -> None:
    """The prewarmed (length, segment-count) buckets are exactly those the batches of
    `cfg` over `dataset`'s captions and images reach (`planned_text` /
    `planned_image`: the minima the prewarm plans with, by default the dataset's),
    every call within MAX_TOKENS."""
    text = 0 if dataset is None else dataset.text_tokens_min()
    image = 0 if dataset is None else dataset.image_tokens_min()
    planned = text if planned_text is None else planned_text
    planned_img = image if planned_image is None else planned_image
    sample, batch = planned_lengths(cfg)
    shortest, segments = planned_segments(cfg, planned, planned_img)
    seen = _attention_calls(prewarm, sample, segments, batch, shortest)
    true_short, true_most = planned_segments(cfg, text, image)
    for length in attention_lengths(sample):
        warmed = {segment_bucket(s) for _, at, s in seen if at == length}
        want = _reachable(length, true_most, batch, true_short)
        assert warmed == want, (length, warmed, want)
    assert all(n <= batch for n, _, _ in seen)


def _fixed_buckets(fn, device, dtype, heads, head_dim, max_len, **kw):
    """The broken variant: every length at 1, 2, 4 and 8 segments of its own length."""
    kw.update(segments=8, shortest=0, max_tokens=0)
    prewarm_attention(fn, device, dtype, heads, head_dim, max_len, **kw)


def test_attention_buckets_follow_the_plan():
    """MAX_TOKENS 16000, 1 MP images (the smallest planned grid 3942 tokens, plus the
    caption-free text): at most 4 samples per batch, so the 8-segment bucket is never
    warmed."""
    cfg = DataConfig(target_area=1024 * 1024, max_tokens=16000, max_text_tokens=512)
    shortest, segments = planned_segments(cfg)
    assert (shortest, segments) == (3942 + cfg.text_overhead_tokens, 4)
    check_attention_buckets(prewarm_attention, cfg)
    count = DataConfig(target_area=512 * 512, batching="count", batch_size=3)
    assert planned_segments(count)[1] == 3
    check_attention_buckets(prewarm_attention, count)
    capped = DataConfig(target_area=256 * 256, max_tokens=20000, max_samples=2)
    assert planned_segments(capped)[1] == 2
    must_fail(lambda: check_attention_buckets(_fixed_buckets, cfg), "fixed 1/2/4/8")


def test_attention_buckets_follow_the_captions():
    """MAX_TOKENS 20000 at 1 MP: captions of 600 tokens at least make the shortest
    sample 4542 tokens, so a batch holds 4 and the 8-segment bucket is never warmed;
    the caption-free bound (3954 tokens, 5 samples) would warm it."""
    cfg = DataConfig(target_area=1024 * 1024, max_tokens=20000, max_text_tokens=1024)
    long = _Captions(600)
    assert planned_segments(cfg, long.text_tokens_min()) == (3942 + 600, 4)
    check_attention_buckets(prewarm_attention, cfg, long)
    must_fail(
        lambda: check_attention_buckets(prewarm_attention, cfg, long, planned_text=0),
        "caption-free bound on long captions",
    )


def test_attention_buckets_follow_the_images():
    """MAX_TOKENS 20000 at 1 MP with images kept below the area (UPSCALE="never"): a
    smallest grid of 256 tokens and captions of 100 make the shortest sample 356
    tokens, so a batch whose longest sample is 4096 tokens holds 8 and more; the area
    target's grid (3942 tokens) would warm 4 at most there."""
    cfg = DataConfig(target_area=1024 * 1024, max_tokens=20000, max_text_tokens=1024)
    small = _Captions(100, image=256)
    assert planned_segments(cfg, 100, 256) == (356, 20000 // 356)
    check_attention_buckets(prewarm_attention, cfg, small)
    must_fail(
        lambda: check_attention_buckets(prewarm_attention, cfg, small, planned_image=0),
        "area-target grid on small images",
    )


def _recorded(joint_grad: bool, text_grad: bool) -> list:
    """The recorded attention calls: the joint one and the text refiner's."""

    def call(rows: int, grad: bool) -> Call:
        q = Snap((rows, 2, 8), torch.float16, torch.device("cpu"), grad)
        return Call(SimpleNamespace(op="attention_varlen"), (q, q, q), {})

    return [call(RECORD_JOINT, joint_grad), call(RECORD_TXT, text_grad)]


def _warmed_keys(warm: ModelPrewarm, calls: list) -> set:
    """(length bucket, segment-count bucket, backward) of every stream call."""
    seen = set()

    def spy(q, k, v, cu, length, scale):
        backward = torch.is_grad_enabled() and q.requires_grad
        seen.add((pow2(length, 64), segment_bucket(cu.shape[0] - 1), backward))
        return q

    for s in warm.streams(calls):
        prewarm_attention(
            spy, "cpu", torch.float32, 1, 8, s.longest, segments=warm.segments,
            backward=s.backward, max_tokens=warm.batch_max, shortest=s.shortest,
        )  # fmt: skip
    return seen


def check_streams(text_max: int) -> None:
    """The live keys of a LoKr run on small images: the frozen text refiner's
    forward-only L128-S8 and the joint L4096-S8 with its backward."""
    warm = ModelPrewarm(
        None, None, None, torch.float16, 5120, 20000, segments=56, shortest=356,
        text_max=text_max, text_shortest=40,
    )  # fmt: skip
    seen = _warmed_keys(warm, _recorded(joint_grad=True, text_grad=False))
    want = {(128, 8, False), (4096, 8, True)}
    assert want <= seen, sorted(want - seen)


def test_attention_streams_cover_the_text_refiner():
    check_streams(text_max=300)
    must_fail(lambda: check_streams(text_max=0), "the text refiner's stream unwarmed")


def _catalog_rows(drop: float):
    """Three rows: a live 120-token caption on a 400-token grid, a tombstoned 20-token
    caption on a 64-token grid, a live 200-token caption on a 900-token grid; caption
    dropout `drop`, the empty prompt 30 tokens."""
    rows = SimpleNamespace(
        _text_est=np.array([120, 20, 200]),
        catalog=SimpleNamespace(live=np.array([1, 0, 1])),
        cfg=SimpleNamespace(text_drop=drop),
        estimate_text_tokens=lambda caption: 30 + len(caption),
        planner=SimpleNamespace(targets=[(20, 20), (8, 8), (30, 30)]),
        geometry=SimpleNamespace(tokens=lambda hw: hw[0] * hw[1]),
    )
    rows._live_text = lambda: MossDataset._live_text(rows)
    return rows


def check_live_minima(text_min, image_min) -> None:
    """The minima the prewarm plans with: live rows only, and the dropped caption's
    length whenever caption dropout is on."""
    assert text_min(_catalog_rows(0.0)) == 120, "a tombstoned caption"
    assert text_min(_catalog_rows(0.1)) == 30, "the dropped caption's bucket"
    assert image_min(_catalog_rows(0.1)) == 400, "a tombstoned row's grid"


def _all_rows_text(ds) -> int:
    """The broken variant: every row, no dropped caption."""
    return int(ds._text_est.min())


def _all_rows_image(ds) -> int:
    """The broken variant: every row's grid."""
    return min(map(ds.geometry.tokens, ds.planner.targets))


def test_planned_minima_follow_the_live_rows():
    check_live_minima(MossDataset.text_tokens_min, MossDataset.image_tokens_min)
    must_fail(
        lambda: check_live_minima(_all_rows_text, MossDataset.image_tokens_min),
        "tombstoned captions, no dropped caption",
    )
    must_fail(
        lambda: check_live_minima(MossDataset.text_tokens_min, _all_rows_image),
        "tombstoned grids",
    )


def _reachable_means(length, segments, max_tokens, shortest) -> set[tuple[int, int]]:
    """(segment-count bucket, mean-length bucket) of every pack a batch can hold: one
    `length` segment plus s - 1 of `shortest` to `length` tokens, `max_tokens` total."""
    short = min(shortest, length)
    out = set()
    for s in range(1, segments + 1):
        lo = length + (s - 1) * short
        hi = min(length + (s - 1) * length, max_tokens)
        if lo > max_tokens:
            break
        for total in range(lo, hi + 1, max(1, (hi - lo) // 4096)):
            out.add((segment_bucket(s), mean_bucket(total, s)))
        out.add((segment_bucket(s), mean_bucket(hi, s)))
    return out


def check_mean_buckets(packs, cfg, dataset) -> None:
    """Every (segment, mean) bucket a batch of `cfg` reaches is prewarmed (`packs`: the
    pack sizes per longest-segment bucket)."""
    sample, batch = planned_lengths(cfg)
    shortest, segments = planned_segments(
        cfg, dataset.text_tokens_min(), dataset.image_tokens_min()
    )
    for length in attention_lengths(sample):
        warmed = {
            (segment_bucket(len(p)), mean_bucket(sum(p), len(p)))
            for p in packs(length, segments, batch, shortest)
        }
        want = _reachable_means(length, segments, batch, shortest)
        assert want <= warmed, (length, sorted(want - warmed))


def _shortest_tails(length, segments, max_tokens, shortest) -> list[list[int]]:
    """The broken variant: every tail at the shortest sample."""
    short = min(shortest or length, length)
    counts = segment_counts(length, segments, max_tokens, shortest)
    return [[length] + [short] * (s - 1) for s in counts]


def test_attention_mean_buckets_follow_the_plan():
    """GBC-like LoKr batches: 36864 tokens of up to 6144-token samples down to 356."""
    cfg = DataConfig(target_area=1024 * 1024, max_tokens=36864, max_text_tokens=2048)
    small = _Captions(100, image=256)
    check_mean_buckets(pack_sizes, cfg, small)
    must_fail(
        lambda: check_mean_buckets(_shortest_tails, cfg, small), "shortest tails only"
    )


def test_segment_counts():
    assert segment_counts(300, 8, 0, 0) == [1, 2, 4, 8]
    assert segment_counts(128, 8, 400, 128) == [1, 2, 3]
    assert segment_counts(300, 8, 400, 50) == [1, 2, 3]
    assert segment_counts(64, 5, 0, 0) == [1, 2, 4, 5]


def test_prewarm_devices():
    cuda = torch.device("cuda", 0)
    assert prewarm_devices(cuda) == [cuda]
    assert prewarm_devices(torch.device("xpu", 1)) == [torch.device("xpu", 1)]
    assert prewarm_devices(torch.device("cpu")) == []


def check_adapter_devices(devices) -> None:
    """Threaded: every stage device; any other kind: the training device; CPU: none."""
    k = {"PARALLEL": "threaded", "DEVICES": (0, 2, 3)}
    cuda = torch.device("cuda", 0)
    stages = [torch.device("cuda", i) for i in (0, 2, 3)]
    assert devices(k, cuda) == stages
    xpu = torch.device("xpu", 1)
    assert devices({**k, "DEVICES": (1, 0)}, xpu) == [xpu, torch.device("xpu", 0)]
    for kind in ("none", "ddp", "fsdp", "pipeline"):
        assert devices({**k, "PARALLEL": kind, "DEVICES": (1,)}, cuda) == [cuda]
    assert devices(k, torch.device("cpu")) == []


def _training_device_only(k: dict, device: torch.device) -> list[torch.device]:
    return prewarm_devices(device)


def test_adapter_prewarm_devices():
    check_adapter_devices(adapter_prewarm_devices)
    must_fail(
        lambda: check_adapter_devices(_training_device_only), "training device only"
    )
