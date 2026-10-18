"""The `auto` planner's micro-step takes the dataset's longest caption as text tokens
per sample (MAX_TEXT_TOKENS while unknown, never a fixed guess) and the OPTIMIZER's
state bytes (model-free: knob dicts, the analytic estimate)."""

import pytest
import torch

from kohakumoss.training.build import model as model_build
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu


def _txt(k: dict, text_tokens) -> int:
    return model_build.workload(k, torch.bfloat16, 1, text_tokens).txt_tokens


def check_caption_lengths(k: dict) -> None:
    short, long = _txt(k, 40), _txt(k, 500)
    assert short < long, (short, long)
    assert _txt(k, None) == _txt(k, k["MAX_TEXT_TOKENS"])


def test_text_tokens_follow_the_dataset(monkeypatch):
    k = runnable(BATCHING="count", BATCH_SIZE=2)
    check_caption_lengths(k)
    assert _txt(k, 40) == 2 * 40
    real = model_build.workload

    def fixed(k, compute, world=1, text_tokens=None):
        return real(k, compute, world, 300)

    monkeypatch.setattr(model_build, "workload", fixed)
    must_fail(lambda: check_caption_lengths(k), "a fixed 300 tokens per caption")


def test_state_bytes_follow_the_optimizer():
    def state(opt: str) -> int:
        k = runnable(OPTIMIZER=opt)
        return model_build.workload(k, torch.bfloat16).optim_state_bytes

    assert state("adamw") == 8 and state("adamw8bit_kernel") == 2
    must_fail(lambda: _equal(state("adamw8bit"), 8), "8-bit states counted as fp32")


def _equal(a, b) -> None:
    assert a == b, (a, b)
