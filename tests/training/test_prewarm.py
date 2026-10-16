"""Kernel prewarm: planned length buckets, the bound action, and binding on CUDA."""

import math

import pytest
import torch

from kohakumoss.model import random_backbone
from kohakumoss.training.build.kernels import (
    BLOCK_OPS,
    Prewarm,
    RetuneKernels,
    block_lengths,
    build_prewarm,
    image_tokens_max,
    planned_lengths,
    prewarm_devices,
    retune_callbacks,
)
from kohakumoss_data import DataConfig
from kohakumoss_kernels.backends.compile import race
from kohakumoss_kernels.tuning import tuner
from tests._util.negative import must_fail


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


@pytest.mark.cpu
@pytest.mark.parametrize("area", [1024 * 1024, 1000 * 1000, 768 * 768, 256 * 256])
def test_image_tokens_max(area):
    cfg = DataConfig(target_area=area, aspect_range=(0.5, 2.0))
    want = _ref_tokens(area, (0.5, 2.0, 1.0), cfg.size_step, cfg.latent_factor)
    assert image_tokens_max(cfg) == want

    def floored():
        assert _floored_tokens(cfg) == want

    must_fail(floored, "sides rounded down to the size step")


@pytest.mark.cpu
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


@pytest.mark.cpu
def test_block_lengths_cover_every_bucket():
    assert block_lengths(16384) == [[256 << i] for i in range(7)]
    assert block_lengths(1000)[-1] == [1024]
    assert block_lengths(100) == [[256]]
    must_fail(
        lambda: _assert_covers(block_lengths(16384)[:-1], 16384), "top bucket missing"
    )
    _assert_covers(block_lengths(9000), 9000)


def _assert_covers(lengths, batch_max) -> None:
    sizes = {sum(x) for x in lengths}
    n = 256
    while True:
        assert n in sizes, n
        if n >= batch_max:
            return
        n *= 2


class _Spy:
    def __init__(self, ret=0):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kw):
        self.calls.append((args, kw))
        return self.ret


@pytest.mark.cpu
def test_prewarm_calls_every_device_with_planned_lengths():
    model = random_backbone("tiny", device="cpu", dtype=torch.float32, seed=0)
    pw = Prewarm(model.ops, model.cfg, torch.float16, 640, 3000)
    pw.attention, pw.blocks, pw.broadcast = _Spy(), _Spy(ret=5), _Spy()
    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    assert pw(devs) == 10
    assert [c[0][1] for c in pw.attention.calls] == devs
    for args, _ in pw.attention.calls:
        assert args[0] is model.ops.attention_varlen
        assert args[3:] == (model.cfg.heads, model.cfg.head_dim, 640)
    for args, _ in pw.blocks.calls:
        assert set(args[0]) == set(BLOCK_OPS)
        assert args[0]["situ_glu"] is model.ops.situ_glu
        _assert_covers(args[3], 3000)
        assert args[4:] == (model.cfg.dim, model.cfg.heads, model.cfg.ffn_dim)
    assert pw.broadcast.calls == []


@pytest.mark.cpu
def test_tune_off_binds_nothing():
    model = random_backbone("tiny", device="cpu", dtype=torch.float32, seed=0)
    k = {"TUNE": "off", "PARALLEL": "none", "DEVICES": (0,)}
    assert not isinstance(build_prewarm(k, model, torch.float16), Prewarm)
    assert retune_callbacks(k) == []
    on = {**k, "TUNE": "auto", "MAX_TOKENS": 2048}
    assert isinstance(build_prewarm(on, model, torch.float16), Prewarm)
    assert isinstance(retune_callbacks(on)[0], RetuneKernels)


@pytest.mark.cpu
def test_prewarm_devices():
    k = {"PARALLEL": "threaded", "DEVICES": (0, 2)}
    cuda = torch.device("cuda", 0)
    assert prewarm_devices(k, cuda) == [
        torch.device("cuda", 0),
        torch.device("cuda", 2),
    ]
    assert prewarm_devices({**k, "PARALLEL": "ddp"}, cuda) == [cuda]
    assert prewarm_devices(k, torch.device("cpu")) == []


@pytest.mark.cuda
def test_prewarm_binds_raced_ops(tmp_path, monkeypatch):
    """After prewarm, a checkpointed forward + backward of a planned length makes no
    kernel choice: no raced op queues a problem and the recompute matches."""
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path))
    mode = tuner.get_mode()
    tuner.set_mode("auto")
    race.reset()
    try:
        dev = torch.device("cuda")
        model = random_backbone("tiny", device=dev, dtype=torch.float16, seed=0)
        k = {"TUNE": "auto", "TARGET_AREA": 256 * 256, "MAX_TOKENS": 1024}
        k.update(PARALLEL="none", DEVICES=(0,), MAX_TEXT_TOKENS=64)
        build_prewarm(k, model, torch.float16)([dev])
        raced = [
            getattr(model.ops, n)
            for n in BLOCK_OPS
            if isinstance(getattr(model.ops, n), race.RacedOp)
        ]
        if not raced:
            pytest.skip("no raced block ops on this device")
        for op in raced:
            assert not op.pending, op.name
            assert tuner.table(dev, op.name).rows, op.name
    finally:
        tuner.set_mode(mode)
        race.reset()
