"""`kohakumoss-tune` records under the autocast the run it tunes for uses: the
trainer's for training shapes, the sampler's for inference shapes, so the stored keys
carry the dtypes those calls meet."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.cli import tune
from kohakumoss.inference.pipeline import sampling_autocast
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
CPU = torch.device("cpu")


def _state(factory) -> tuple[bool, torch.dtype | None]:
    with factory():
        on = torch.is_autocast_enabled("cpu")
        return on, torch.get_autocast_dtype("cpu") if on else None


def _backbone(dtype: torch.dtype):
    """A stand-in carrying only the joint-block weight dtype `compute_dtype` reads."""
    q = SimpleNamespace(weight_like=torch.empty(0, dtype=dtype))
    return SimpleNamespace(layers=[SimpleNamespace(proj={"q": q})])


def check_autocast(record_autocast) -> None:
    assert _state(record_autocast("bf16", CPU, True)) == (True, torch.bfloat16)
    sampler = _state(sampling_autocast(_backbone(torch.bfloat16), CPU))
    assert (
        _state(record_autocast("bf16", CPU, False))
        == sampler
        == (
            True,
            torch.bfloat16,
        )
    )


def test_recorded_under_the_runs_autocast():
    check_autocast(tune.record_autocast)


def test_negative_no_autocast():
    must_fail(
        lambda: check_autocast(lambda dtype, device, backward: contextlib.nullcontext),
        "shapes recorded without autocast",
    )


def test_tune_does_not_shadow_precision_autocast_for():
    assert not hasattr(tune, "autocast_for") or tune.autocast_for.__module__ != (
        tune.__name__
    )
