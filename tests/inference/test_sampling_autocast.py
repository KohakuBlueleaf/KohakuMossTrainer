"""The sampler's DiT forwards run under training's autocast to the backbone's compute
dtype, so the fp32-stored conditioning linears take 16-bit inputs (model-free: the
context the pipeline opens around sampling, a fp32 linear inside it)."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kohakumoss.inference.pipeline import sampling_autocast
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def backbone(dtype: torch.dtype):
    """The attribute compute_dtype reads (the joint blocks' weight dtype)."""
    q = SimpleNamespace(weight_like=torch.empty(0, dtype=dtype))
    return SimpleNamespace(layers=[SimpleNamespace(proj={"q": q})])


def _sixteen_bit_inputs(context) -> None:
    x, w = torch.randn(4, 8), torch.randn(8, 8)
    with context():
        y = F.linear(x, w)
    assert y.dtype == torch.bfloat16, y.dtype


def test_the_conditioning_linears_see_16_bit_inputs():
    _sixteen_bit_inputs(sampling_autocast(backbone(torch.bfloat16), "cpu"))
    assert sampling_autocast(backbone(torch.float32), "cpu") is contextlib.nullcontext
    must_fail(lambda: _sixteen_bit_inputs(contextlib.nullcontext), "no autocast")
