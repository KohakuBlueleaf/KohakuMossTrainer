"""Per-step draws: DyLoRA ranks and module-dropout skips are drawn once per microbatch
from (seed, step, microbatch), so checkpoint recomputes and every rank agree."""

import random

import pytest
import torch
from _adcases import build
from _adtoy import inputs
from lycoris.modules import DyLoraModule
from torch.utils.checkpoint import checkpoint

from kohakumoss.adapters.bind import eager
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _checkpointed_steps(model, aset, monkeypatch, steps: int = 20) -> None:
    """Every layer's forward and its checkpoint recompute use the same rank."""
    used: dict[int, list[int]] = {}
    real = DyLoraModule.get_weight

    def logged(self, rank):
        out = real(self, rank)
        used.setdefault(id(self), []).append(out[0].shape[0])
        return out

    monkeypatch.setattr(DyLoraModule, "get_weight", logged)
    latent, text = inputs()
    model.train()
    for step in range(steps):
        used.clear()
        aset.begin_step(torch.rand(1), None, step)
        y = checkpoint(model, latent, text, use_reentrant=False)
        y.square().sum().backward()
        aset.end_step()
        for ranks in used.values():
            assert len(ranks) == 2 and ranks[0] == ranks[1], ranks


def test_dylora_trains_under_activation_checkpointing(monkeypatch):
    model, aset = build("dylora", dim=8)
    _checkpointed_steps(model, aset, monkeypatch)
    ranks = {t.draws.rank for t in aset.targets}
    assert ranks <= {2, 4, 6, 8} and len(ranks) > 1


def _random_draw(linear, lyco, multiplier, draws, dense, x):
    rank = random.randint(1, lyco.block_count) * lyco.block_size
    dw = lyco.get_diff_weight(multiplier, rank=rank)[0].reshape(lyco.shape)
    return linear._impl(x) + torch.nn.functional.linear(x, dw.to(x.dtype))


def test_a_per_call_rank_draw_breaks_the_recompute(monkeypatch):
    def per_call():
        with monkeypatch.context() as patch:
            patch.setattr(eager, "dylora_train", _random_draw)
            model, aset = build("dylora", dim=8)
            try:
                _checkpointed_steps(model, aset, patch)
            except RuntimeError as e:
                raise AssertionError(e) from e

    must_fail(per_call, "a per-call rank from Python's random under checkpoint")


def _draws(aset, step: int, seed_noise: int):
    torch.manual_seed(seed_noise)
    aset.begin_step(torch.rand(1), None, step)
    out = [(t.draws.skip, t.draws.rank) for t in aset.targets]
    return out


def test_draws_depend_only_on_seed_step_and_microbatch():
    _, a = build("dylora", module_dropout=0.5, seed=0)
    _, b = build("dylora", module_dropout=0.5, seed=1)
    for aset in (a, b):
        aset.targets[0].lyco.train()
    first = _draws(a, 3, 11)
    assert first == _draws(b, 3, 99)
    second = _draws(a, 3, 5)
    assert second == _draws(b, 3, 7)
    assert second != first
    a.end_step()
    assert _draws(a, 3, 1) == first
    assert _draws(a, 4, 1) != first


def test_end_step_clears_skips():
    _, aset = build("lora", module_dropout=0.9)
    aset.begin_step(torch.rand(1), None, 0)
    assert any(t.draws.skip for t in aset.targets)
    aset.end_step()
    assert not any(t.draws.skip for t in aset.targets)
