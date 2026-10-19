"""Stored bases under the base precision (model-free, MossLinear holders): `auto` and
the stored format keep a pre-quantized file's bases, any other precision dequantizes
them to the exact dense weight, a trainable base under `auto` is refused; the memory
estimate counts the stored bytes under `auto`; a text model's quantized modules go back
to nn.Linear / nn.Embedding."""

import pytest
import torch
from torch import nn

from kohakumoss.memory.apply import settle_stored
from kohakumoss.memory.estimate import base_precision, weight_bytes
from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.densify import densify_text, stored_formats, stored_precision_of
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.weights import QuantWeight
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _holder(rotation=256):
    g = torch.Generator().manual_seed(0)
    lin = MossLinear(256, 32, bias=False, dtype=torch.bfloat16)
    q = torch.randint(-127, 128, (32, 256), generator=g, dtype=torch.int8)
    base = QuantWeight(q, torch.rand(32, generator=g) * 1e-2, "int8", rotation)
    lin.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
    lin.set_base(base)
    return nn.ModuleDict({"lin": lin}), base


def test_auto_and_the_stored_format_keep_the_bases():
    for precision in ("auto", "int8"):
        holder, base = _holder()
        settle_stored(holder, precision, trainable_base=False)
        assert holder["lin"].base is base, precision
    assert stored_precision_of(holder) == "int8"


def check_dense(holder, want) -> None:
    lin = holder["lin"]
    assert lin.base is None and lin.weight.dtype == torch.bfloat16
    assert torch.equal(lin.weight, want)


def test_other_precisions_dequantize_exactly():
    for precision, trainable in (("compute", False), ("fp8", False), ("compute", True)):
        holder, base = _holder()
        want = base.weight(torch.bfloat16)
        settle_stored(holder, precision, trainable_base=trainable)
        check_dense(holder, want)
        assert stored_formats(holder) == set()
    holder, base = _holder()
    rotated = base.weight(torch.bfloat16)
    settle_stored(holder, "compute", trainable_base=False)
    plain = QuantWeight(base.q, base.scale, "int8").weight(torch.bfloat16)
    must_fail(lambda: check_dense(holder, plain), "the rotation not undone")
    check_dense(holder, rotated)


def test_trainable_auto_is_refused():
    holder, _ = _holder()
    with pytest.raises(ValueError, match="BASE_PRECISION='compute'"):
        settle_stored(holder, "auto", trainable_base=True)


def test_estimate_counts_the_stored_bytes_under_auto():
    auto = MemoryPlan(base_precision="auto")
    assert base_precision(auto, "int8") == "int8"
    assert weight_bytes(auto, 2, training=True, stored="int8") == 1
    assert weight_bytes(auto, 2, training=True) == 2
    assert weight_bytes(MemoryPlan(base_precision="compute"), 2, True, stored="int8") == 2  # fmt: skip
    must_fail(lambda: _assert_bytes(auto, "int8", 2), "auto counted 16-bit")


def _assert_bytes(plan, stored, want) -> None:
    assert weight_bytes(plan, 2, training=True, stored=stored) == want


def test_text_modules_densify():
    g = torch.Generator().manual_seed(1)
    q = torch.randint(-127, 128, (64, 256), generator=g, dtype=torch.int8)
    scale = torch.rand(64, generator=g) * 1e-2
    model = nn.Module()
    model.proj = QuantLinear.from_base(
        256, 64, QuantWeight(q, scale, "int8", 256), None, torch.float32
    )
    model.embed = QuantEmbedding(q, scale, 256, torch.float32)
    want = QuantWeight(q, scale, "int8", 256).weight(torch.float32)
    assert densify_text(model, torch.float32) == 2
    assert type(model.proj) is nn.Linear and type(model.embed) is nn.Embedding
    assert torch.equal(model.proj.weight, want) and torch.equal(
        model.embed.weight, want
    )
