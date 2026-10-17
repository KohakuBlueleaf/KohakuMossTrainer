"""Online caches in the trainer: per-batch hit / miss, text_limit, full-grid write-back,
writer flush at checkpoints, and the data knobs of pipeline layouts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.training.callbacks import CallbackRunner
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers.native import NativeDriver
from kohakumoss.training.entry import run
from kohakumoss.training.knobs import validate
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import (
    DeviceBatcher,
    OnlineLatents,
    OnlineText,
    gather_kept,
)
from kohakumoss.types.step import StepOut
from kohakumoss_data import CacheWriter, DataConfig, MossDataset
from kohakumoss_data.cache.random_encoders import RandomImageEncoder, RandomTextEncoder
from kohakumoss_data.cache.store import TensorStore
from tests._util.negative import must_fail
from tests.training._train_fakes import host_batch
from tests.training.test_entry_smoke import (
    assert_same,
    assert_trained,
    exact_runs,
    smoke_knobs,
    trainable_at,
)
from tests.training.test_knobs import runnable

pytestmark = pytest.mark.cpu
DEV = torch.device("cpu")
ENC_IMG = RandomImageEncoder(channels=16)
ENC_TXT = RandomTextEncoder(dim=32)


class Counting:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.fn(*args)


def _online_batcher(img, txt, writer=None) -> DeviceBatcher:
    kw = {} if writer is None else {"on_encoded": writer}
    return DeviceBatcher(DEV, text=OnlineText(txt), latents=OnlineLatents(img), **kw)


def _hit_uses_batch(hb) -> None:
    img, txt = Counting(ENC_IMG), Counting(ENC_TXT)
    batch = _online_batcher(img, txt)(hb)
    assert img.calls == 0 and txt.calls == 0
    assert torch.equal(batch.latents, hb.latents) and torch.equal(batch.text, hb.text)


def test_cache_hit_is_used_without_encoding():
    hb = host_batch(0, 0, 16, 32)
    _hit_uses_batch(hb)
    miss = host_batch(0, 0, 16, 32)
    miss.pixels = [np.zeros((h * 16, w * 16, 3), np.uint8) for h, w in miss.hw.tolist()]
    miss.latents = None
    must_fail(lambda: _hit_uses_batch(miss), "a miss encoded like a hit")


def _prompts_batch(limits) -> object:
    hb = host_batch(0, 0, 16, 32)
    hb.text, hb.prompts, hb.text_limit = None, ["a cat on a mat", "dog"], limits
    return hb


def test_online_text_applies_text_limit():
    hb = _prompts_batch([5, 0])
    text, lens = OnlineText(ENC_TXT)(hb, DEV)
    full = ENC_TXT(hb.prompts)
    want = torch.cat([full[0][:5], full[1]])
    assert lens.tolist() == [5, full[1].shape[0]] and torch.equal(text, want)
    uncapped = OnlineText(ENC_TXT)(_prompts_batch([]), DEV)

    def ignored_limit():
        assert torch.equal(uncapped[0], want)

    must_fail(ignored_limit, "text_limit not applied")


class _NullBundle:
    """Optimiser bundle stand-in: steps always succeed, nothing is trained."""

    last_loss, last_norm, last_ok, lr, skipped, nonfinite = 0.0, 0.0, True, 0.0, 0, 0
    scaler = SimpleNamespace(scale=1.0)

    def backward(self, loss) -> None:
        return None

    def step(self) -> bool:
        return True


class _NoFill(OnlineText):
    """OnlineText that leaves `hb.text_len` at the loader's zeros."""

    def __call__(self, hb, device):
        lens = hb.text_len.clone()
        out = super().__call__(hb, device)
        hb.text_len = lens
        return out


def _run_counts(text_fn, steps: int = 3) -> TrainState:
    loader = [_prompts_batch([]) for _ in range(steps)]
    for hb in loader:
        hb.text_len = torch.zeros(2, dtype=torch.int32)
    driver = NativeDriver(
        steps,
        1,
        DeviceBatcher(DEV, text=text_fn),
        TokenNormalizer(),
        DEV,
        CallbackRunner([]),
        SimpleNamespace(save=lambda step, parts: None, saved=lambda step: False),
        lambda state: {},
    )
    ema = SimpleNamespace(update=lambda i: None)
    return driver.run(_null_step, loader, _NullBundle(), ema, TrainState())


def _null_step(batch, denom, step) -> StepOut:
    return StepOut(torch.zeros(()), denom)


def test_online_text_tokens_are_accounted():
    state = _run_counts(OnlineText(ENC_TXT))
    lens = [t.shape[0] for t in ENC_TXT(_prompts_batch([]).prompts)]
    img = [h * w for h, w in _prompts_batch([]).hw.tolist()]
    assert state.text_tokens == 3 * sum(lens)
    assert state.joint_sq == 3 * sum((a + b) ** 2 for a, b in zip(img, lens))

    def unfilled():
        assert _run_counts(_NoFill(ENC_TXT)).text_tokens == 3 * sum(lens)

    must_fail(unfilled, "encoded text lengths not written back to the host batch")


def _drop_cfg(tmp_path) -> DataConfig:
    return DataConfig(
        datasets=[{"source": "synthetic", "count": 24, "seed": 2}],
        resolution_mode="resize",
        batching="tokens",
        max_tokens=300,
        overlong="drop_tokens",
        target_area=256 * 256,
        latent_cache="online",
        text_cache="online",
        cache_dir=str(tmp_path / "cache"),
        caption_variants=1,
        text_drop=0.0,
    )


def test_token_drop_samples_are_cached_at_full_grid(tmp_path):
    ds = MossDataset(_drop_cfg(tmp_path))
    writer = CacheWriter(ds)
    batcher = _online_batcher(ENC_IMG, ENC_TXT, writer)
    seen = []
    for index in range(6):
        hb = ds.materialize(index)
        batch = batcher(hb)
        full = ENC_IMG(hb.pixels, hb.hw)
        assert torch.equal(batch.latents, gather_kept(full, hb.hw, hb.img_keep))
        sizes = (hb.hw[:, 0] * hb.hw[:, 1]).tolist()
        seen += list(zip(hb.latent_keys, full.split(sizes), hb.img_keep))
    writer.close()
    assert any(keep is not None for *_, keep in seen)
    store = TensorStore(ds.latents.store.path)
    for key, full, _ in seen:
        assert torch.equal(store.get(key).to(full.dtype), full), key

    def kept_written():
        key, full, keep = next(s for s in seen if s[2] is not None)
        assert torch.equal(store.get(key).to(full.dtype), full.index_select(0, keep))

    must_fail(kept_written, "kept subset stored as the cache value")


def _hit_vs_miss(tmp_path, rounded: bool):
    """(model inputs of a miss, model inputs of the same batch read back as a hit)."""
    cfg = DataConfig(
        datasets=[{"source": "synthetic", "count": 8, "seed": 2}],
        resolution_mode="square_crop",
        batching="count",
        batch_size=2,
        target_area=256 * 256,
        latent_cache="online",
        text_cache="online",
        cache_dtype="fp8",
        cache_dir=str(tmp_path / ("r" if rounded else "raw")),
        caption_variants=1,
        text_drop=0.0,
    )
    ds = MossDataset(cfg)
    kw = (
        {"text_round": ds.texts.as_cached, "latent_round": ds.latents.as_cached}
        if rounded
        else {}
    )
    writer = CacheWriter(ds)
    batcher = DeviceBatcher(
        DEV,
        text=OnlineText(ENC_TXT, **({"as_cached": kw["text_round"]} if kw else {})),
        latents=OnlineLatents(
            ENC_IMG, **({"as_cached": kw["latent_round"]} if kw else {})
        ),
        on_encoded=writer,
    )
    miss_hb = ds.materialize(0)
    assert miss_hb.pixels is not None
    miss = batcher(miss_hb)
    writer.close()
    hit_hb = MossDataset(cfg).materialize(0)
    assert hit_hb.pixels is None
    hit = batcher(hit_hb)
    return miss, hit


def _same_inputs(miss, hit) -> None:
    assert torch.equal(miss.latents.float(), hit.latents.float())
    assert torch.equal(miss.text.float(), hit.text.float())


def test_miss_trains_on_what_a_hit_returns(tmp_path):
    _same_inputs(*_hit_vs_miss(tmp_path, rounded=True))
    must_fail(
        lambda: _same_inputs(*_hit_vs_miss(tmp_path, rounded=False)),
        "fresh encodings not rounded like the cache",
    )


def _online_run(out, cache, **kw) -> None:
    run(
        smoke_knobs(
            out,
            LATENT_CACHE="online",
            TEXT_CACHE="online",
            CACHE_DIR=str(cache),
            CACHE_DTYPE="fp8",
            TEXT_DROP=0.0,
            **kw,
        )
    )


def test_online_cache_resume_equals_uninterrupted(tmp_path):
    same = exact_runs()
    _online_run(tmp_path / "full", tmp_path / "c_full", STEPS=4, SAVE_EVERY=0, **same)
    _online_run(tmp_path / "part", tmp_path / "c_part", STEPS=2, SAVE_EVERY=2, **same)
    _online_run(
        tmp_path / "part",
        tmp_path / "c_part",
        STEPS=4,
        SAVE_EVERY=2,
        RESUME="latest",
        **same,
    )
    full = trainable_at(tmp_path / "full", 4)
    resumed = trainable_at(tmp_path / "part", 4)
    assert_trained(full)
    assert_same(full, resumed)
    must_fail(
        lambda: assert_same(full, trainable_at(tmp_path / "part", 2)),
        "compared against an earlier step",
    )


def test_checkpoint_save_flushes_first(tmp_path):
    order = []
    ckpt = CheckpointManager(
        tmp_path, 1, is_main=False, before_save=lambda: order.append("flush")
    )
    ckpt.save(1, {"step_rng": {}})
    ckpt.save(2, {"step_rng": {}})
    assert order == ["flush", "flush"]


def test_entry_online_caches_write_back(tmp_path):
    cache = tmp_path / "cache"
    knobs = smoke_knobs(
        tmp_path,
        LATENT_CACHE="online",
        TEXT_CACHE="online",
        CACHE_DIR=str(cache),
        STEPS=2,
        TEXT_DROP=0.0,
    )
    run(knobs)
    cfg = DataConfig.from_knobs(knobs)
    ds = MossDataset(cfg)
    hb = ds.materialize(0)
    assert hb.pixels is None and hb.prompts is None, "batch 0 was not written back"
    assert ds.materialize(1).pixels is None, "batch 1 was not written back"


def test_data_knobs():
    k = runnable(MAX_SAMPLES=-1, TOKEN_BUDGET="exactly")
    problems = " ".join(validate(k))
    assert "MAX_SAMPLES" in problems and "TOKEN_BUDGET" in problems
    assert validate(runnable(MAX_SAMPLES=4, TOKEN_BUDGET="estimate")) == []


@pytest.mark.parametrize(
    "fix, missing",
    [
        ({}, 4),
        ({"BATCHING": "tokens"}, 3),
        ({"BATCHING": "tokens", "MAX_SAMPLES": 2}, 2),
        (
            {
                "BATCHING": "tokens",
                "MAX_SAMPLES": 2,
                "TOKEN_BUDGET": "exact",
                "TEXT_CACHE": "offline",
            },
            0,
        ),
    ],
)
def test_pipeline_data_rules(fix, missing):
    base = dict(PARALLEL="pipeline", DEVICES=(0, 1), MICROBATCHES=2, BATCHING="count")
    problems = validate(runnable(**{**base, "TEXT_CACHE": "off", **fix}))
    data = [p for p in problems if "pipeline needs" in p or "TOKEN_BUDGET" in p]
    assert len(data) == missing, data
