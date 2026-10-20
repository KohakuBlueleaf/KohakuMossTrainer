"""Quantization from a weight's stored values (`kohakumoss.quant.source`), model-free on
raw tensors: a weight stored in bf16 and live as its fp16 cast quantizes from the bf16
values, so its codes do not depend on the compute dtype; a live weight that is not the
cast of the stored one (a merged delta) quantizes as it is; a fold applied since the
load is multiplied into the stored values; the same through `quantize_model` (DiT) and
`quantize_text_linears` (text encoder); a weight the load left unread is quantized
from the checkpoint, and the unread set is the quantized one."""

import logging

import pytest
import torch
from torch import nn

from kohakumoss.memory.apply import quantize_base, quantized_weights
from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.convert import quantize_model, unread_problems
from kohakumoss.quant.source import (
    DEFERRED,
    NO_SOURCE,
    NOT_THE_CAST,
    SAME,
    SOURCES,
    STORED,
    SourceWeights,
    source_weight,
    suffix_resolver,
)
from kohakumoss.quant.text import quantize_text_linears
from kohakumoss.quant.weights import quantize
from kohakumoss.types.plans import MemoryPlan
from kohakumoss_kernels.fp8 import quantize as fp8_quantize
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu

OUT, IN = 256, 512
FP16_NORMAL = torch.finfo(torch.float16).tiny
FORMATS = ("fp8", "row", "mx")


class _Stored:
    """A checkpoint's tensors by key (keys / dtype / get), counting reads."""

    def __init__(self, state: dict[str, torch.Tensor]):
        self.state = state
        self.reads = 0

    def keys(self) -> list[str]:
        return list(self.state)

    def dtype(self, key: str) -> torch.dtype:
        return self.state[key].dtype

    def get(self, key: str) -> torch.Tensor:
        self.reads += 1
        return self.state[key]


def _source(state: dict[str, torch.Tensor]) -> SourceWeights:
    tensors = _Stored(state)
    return SourceWeights(tensors, lambda name: name if name in state else None)


def stored_weight(seed: int = 0) -> torch.Tensor:
    """A bf16 weight with a tenth of its entries below fp16's smallest normal
    (log-uniform down to 1e-9, where the fp16 cast rounds them)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT, IN, generator=g) * 0.02
    tiny = torch.rand(OUT, IN, generator=g) < 0.1
    exponent = torch.rand(OUT, IN, generator=g) * (
        9 + torch.log10(torch.tensor(FP16_NORMAL))
    )
    w = torch.where(tiny, 10 ** (exponent - 9), w)
    return w.to(torch.bfloat16)


def codes(w: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes as bytes, scale) of `w` under a weight-only fp8 or an fp8-compute recipe."""
    q, scale = quantize(w, "fp8") if fmt == "fp8" else fp8_quantize(w, fmt, "weight")
    return q.view(torch.uint8), scale


def check_codes(values: torch.Tensor, stored: torch.Tensor, fmt: str) -> None:
    """`values` quantize to the stored weight's own codes and scales."""
    q, s = codes(values, fmt)
    q_ref, s_ref = codes(stored, fmt)
    assert torch.equal(s, s_ref), fmt
    assert torch.equal(q, q_ref), (fmt, int((q != q_ref).sum()))


@pytest.mark.parametrize("fmt", FORMATS)
def test_codes_follow_the_stored_values(fmt):
    w = stored_weight()
    live = w.to(torch.float16)
    values, kind = source_weight(live, _source({"a.weight": w}), "a.weight")
    assert kind == STORED, kind
    check_codes(values, w, fmt)
    must_fail(lambda: check_codes(live, w, fmt), "quantizing the live fp16 cast")


def check_kept(values: torch.Tensor, kind: str, live: torch.Tensor) -> None:
    """A live weight that is not the stored one's cast is quantized as it is."""
    assert kind == NOT_THE_CAST, kind
    assert torch.equal(values, live)


def test_a_live_weight_not_the_cast_is_kept():
    w = stored_weight()
    live = (w.float() + 1e-3).to(torch.float16)
    values, kind = source_weight(live, _source({"a.weight": w}), "a.weight")
    check_kept(values, kind, live)
    must_fail(
        lambda: check_kept(w.float(), STORED, live),
        "the stored values without the guard",
    )


def test_a_fold_since_the_load_is_multiplied_in():
    w = stored_weight()
    rows = torch.where(torch.arange(OUT) < OUT // 2, 0.5, 1.0)
    live = (w.float() * rows[:, None]).to(torch.float16)
    source = _source({"a.weight": w})
    values, kind = source_weight(live, source, "a.weight", rows)
    assert kind == STORED, kind
    assert torch.equal(values, w.float() * rows[:, None])
    unscaled, kind = source_weight(live, source, "a.weight")
    assert kind == NOT_THE_CAST, kind
    must_fail(
        lambda: check_codes(unscaled, w.float() * rows[:, None], "fp8"), "no fold"
    )


def check_no_read(tensors: _Stored, kind: str) -> None:
    assert tensors.reads == 0 and kind == SAME, (tensors.reads, kind)


def test_same_dtype_reads_nothing():
    w = stored_weight()
    tensors = _Stored({"a.weight": w})
    source = SourceWeights(tensors, lambda name: name)
    _, kind = source_weight(w, source, "a.weight")
    check_no_read(tensors, kind)
    _, kind = source_weight(w.to(torch.float16), source, "a.weight")
    must_fail(lambda: check_no_read(tensors, kind), "another dtype reads it")


def test_no_source_is_named():
    w = stored_weight()
    _, kind = source_weight(w.to(torch.float16), None, "a.weight")
    assert kind == NO_SOURCE, kind
    _, kind = source_weight(w.to(torch.float16), _source({}), "a.weight")
    assert kind == NO_SOURCE, kind
    _, kind = source_weight(w.to(torch.float16), _source({"a.weight": w}), "a.weight")
    must_fail(lambda: check_no_source(kind), "a weight the checkpoint holds")


def check_no_source(kind: str) -> None:
    assert kind == NO_SOURCE, kind


def test_one_info_line_counts_every_kind(caplog):
    w = stored_weight()
    text = nn.ModuleDict({"a": _linear(w)["proj"], "b": _linear(w + 1e-3)["proj"]})
    state = {"a.weight": w, "b.weight": w}
    with caplog.at_level(logging.INFO, logger="kohakumoss.quant.source"):
        report = quantize_text_linears(
            text, "fp8", 0, torch.float16, source=_source(state)
        )
    lines = [
        r.getMessage()
        for r in caplog.records
        if "quantized 2 linears" in r.getMessage()
    ]
    check_tally(report, lines)
    must_fail(
        lambda: check_tally({**report, "sources": {STORED: 2}}, lines),
        "fallbacks hidden",
    )


def check_tally(report: dict, lines: list[str]) -> None:
    assert report["sources"] == {STORED: 1, NOT_THE_CAST: 1}, report["sources"]
    assert len(lines) == 1, lines
    assert SOURCES[STORED] in lines[0] and SOURCES[NOT_THE_CAST] in lines[0], lines


def check_resolves(resolve) -> None:
    assert resolve("layers.0.mlp.down_proj.weight") is None
    assert resolve("layers.0.mlp.up_proj.weight") == (
        "model.language_model.layers.0.mlp.up_proj.weight"
    )
    assert resolve("missing.weight") is None


def test_suffix_resolver_needs_one_match():
    keys = [
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.visual.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    ]
    check_resolves(suffix_resolver(keys))

    def first(name):
        return next((k for k in keys if k.endswith("." + name)), None)

    must_fail(lambda: check_resolves(first), "the first suffix match")


def check_base(base, want: torch.Tensor) -> None:
    """A QuantWeight holds the fp8 codes and scales of `want`."""
    q_ref, s_ref = codes(want, "fp8")
    assert torch.equal(base.scale, s_ref)
    assert torch.equal(base.q.view(torch.uint8), q_ref)


def _moss(w: torch.Tensor) -> MossLinear:
    lin = MossLinear(IN, OUT, bias=False, dtype=torch.float16)
    lin.weight.data.copy_(w.to(torch.float16))
    return lin


def test_quantize_model_takes_the_stored_values():
    w = stored_weight()
    rows = torch.where(torch.arange(OUT) < OUT // 2, 0.5, 1.0)
    want = w.float() * rows[:, None]
    holder = nn.ModuleDict({"lin": _moss(want)})
    report = quantize_model(
        holder, "fp8", source=_source({"lin.weight": w}), scales={"lin": rows}
    )
    assert report["from_source"] == 1, report
    check_base(holder["lin"].base, want)
    old = nn.ModuleDict({"lin": _moss(want)})
    quantize_model(old, "fp8")
    must_fail(lambda: check_base(old["lin"].base, want), "no source given")


def _unread() -> nn.ModuleDict:
    """A linear whose weight the load left unread (meta, `deferred_weights`)."""
    with torch.device("meta"):
        lin = MossLinear(IN, OUT, bias=False, dtype=torch.float16)
    holder = nn.ModuleDict({"lin": lin})
    holder.deferred_weights = {"lin.weight": torch.device("cpu")}
    return holder


def test_a_weight_left_unread_is_quantized_from_the_checkpoint():
    w = stored_weight()
    rows = torch.where(torch.arange(OUT) < OUT // 2, 0.5, 1.0)
    want = w.float() * rows[:, None]
    holder = _unread()
    source = _source({"lin.weight": w})
    report = quantize_model(holder, "fp8", source=source, scales={"lin": rows})
    assert report["sources"] == {DEFERRED: 1} and report["from_source"] == 1, report
    assert holder["lin"].base.q.device.type == "cpu"
    assert not holder["lin"].weight.is_meta and unread_problems(holder) == []
    check_base(holder["lin"].base, want)
    unscaled = _unread()
    quantize_model(unscaled, "fp8", source=source)
    must_fail(lambda: check_base(unscaled["lin"].base, want), "the fold left out")
    missed = _unread()
    quantize_model(missed, "fp8", lambda name, lin: False, source=source)
    assert unread_problems(missed), "an unread weight left unquantized"


def test_the_unread_weights_are_the_quantized_ones():
    plan = MemoryPlan(base_precision="int8")

    def model() -> nn.ModuleDict:
        sizes = {"big": (512, 256), "small": (64, 64), "kept": (512, 256)}
        holder = nn.ModuleDict(
            {
                k: MossLinear(i, o, bias=False, dtype=torch.float16)
                for k, (i, o) in sizes.items()
            }
        )
        holder.weight_path = None
        return holder

    def check(names_of) -> None:
        names = names_of(model())
        built = model()
        quantize_base(built, plan, torch.float16, True, False, ("kept",))
        done = {f"{n}.weight" for n, m in built.items() if m.base is not None}
        assert names == done == {"big.weight"}, (names, done)

    check(lambda m: quantized_weights(m, plan, False, ("kept",)))
    must_fail(
        lambda: check(lambda m: quantized_weights(m, plan, False)), "KEEP_FP32 lost"
    )
    assert quantized_weights(model(), plan, True, ("kept",)) == set()


def _linear(w: torch.Tensor) -> nn.ModuleDict:
    lin = nn.Linear(IN, OUT, bias=False, dtype=torch.float16)
    lin.weight.data.copy_(w.to(torch.float16))
    return nn.ModuleDict({"proj": lin})


def test_text_linears_take_the_stored_values():
    w = stored_weight()
    text = _linear(w)
    report = quantize_text_linears(
        text, "fp8", 0, torch.float16, source=_source({"proj.weight": w})
    )
    assert report["from_source"] == 1, report
    check_base(text["proj"].qweight, w)
    plain = _linear(w)
    quantize_text_linears(plain, "fp8", 0, torch.float16)
    must_fail(lambda: check_base(plain["proj"].qweight, w), "no source given")
