"""The streamed text-encoder load's model-free parts: checkpoint keys to text-model
names, file tensors read by those names in their stored dtype, the quantization source's dtype
filter, `linear_fate` / `quantizes` agreeing with what `quantize_text_linears`
replaces, the meta-device size delta of a quantized load, and the host bytes a copy
adds."""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kohakumoss.encoders.text.files import TextFiles, hf_text_name
from kohakumoss.encoders.text.load import _Fill, copy_bytes, pow2, quantized_delta
from kohakumoss.encoders.text.plan import text_plan
from kohakumoss.quant.text import linear_fate, quantize_text_linears, quantizes
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_hf_text_names():
    assert hf_text_name("language_model.layers.0.mlp.up_proj.weight") == (
        "layers.0.mlp.up_proj.weight"
    )
    assert hf_text_name("model.language_model.norm.weight") == "norm.weight"
    assert hf_text_name("model.layers.1.input_layernorm.weight") == (
        "layers.1.input_layernorm.weight"
    )
    assert hf_text_name("embed_tokens.weight") == "embed_tokens.weight"
    for skipped in (
        "visual.blocks.0.attn.qkv.weight",
        "lm_head.weight",
        "mtp.fc.weight",
    ):
        assert hf_text_name(skipped) is None


def _file(tmp_path, tensors: dict) -> TextFiles:
    path = tmp_path / "te.safetensors"
    save_file(tensors, str(path))
    return TextFiles([path], hf_text_name)


def assert_files(files: TextFiles, want: dict) -> None:
    assert sorted(files.keys()) == sorted(
        n.removeprefix("language_model.") for n in want
    )
    for key, value in want.items():
        got = files.get(key.removeprefix("language_model."))
        assert got.dtype == value.dtype and torch.equal(got, value)


def test_text_files_read_views(tmp_path):
    want = {
        "language_model.norm.weight": torch.arange(8, dtype=torch.bfloat16),
        "language_model.layers.0.w.weight": torch.ones(4, 4, dtype=torch.float16),
        "language_model.layers.0.codes.weight": torch.ones(4, 4, dtype=torch.int8),
        "visual.patch.weight": torch.zeros(2),
    }
    files = _file(tmp_path, want)
    assert_files(files, {k: v for k, v in want.items() if not k.startswith("visual")})
    src = files.source("layers.0.")
    assert src.dtype("w.weight") == torch.float16
    assert src.resolve("codes.weight") is None
    assert files.source("layers.0.", (torch.bfloat16,)).resolve("w.weight") is None
    must_fail(lambda: assert_files(files, {"norm.weight": want[next(iter(want))]}))


def test_text_files_refuse_two_keys_for_one_name(tmp_path):
    tensors = {
        "language_model.norm.weight": torch.ones(2),
        "norm.weight": torch.ones(2),
    }
    with pytest.raises(ValueError, match="both name"):
        _file(tmp_path, tensors)


def _block() -> nn.Module:
    m = nn.Module()
    m.big = nn.Linear(512, 384)
    m.small = nn.Linear(64, 512)
    m.odd = nn.Linear(500, 512)
    m.inner = nn.Sequential(nn.Linear(256, 256, bias=False))
    return m


def assert_fates(fate, precision: str, group: int) -> None:
    m = _block()
    named = {n: c for n, c in m.named_modules() if isinstance(c, nn.Linear)}
    said = {n for n, c in named.items() if fate(c, precision, group) == "quantize"}
    quantize_text_linears(m, precision, group, torch.float32)
    done = {n for n, c in m.named_modules() if n in named and type(c) is not nn.Linear}
    assert said == done
    assert quantizes(_block(), precision, group) == bool(done)


@pytest.mark.parametrize("precision, group", [("int8", 0), ("int8", 128), ("fp8", 0)])
def test_linear_fate_is_what_quantize_replaces(precision, group):
    assert_fates(linear_fate, precision, group)


def _ignores_group(module, precision, group, min_features=256):
    return linear_fate(module, precision, 0, min_features)


def test_fate_check_fails_on_a_broken_fate():
    must_fail(lambda: assert_fates(_ignores_group, "int8", 128), "group ignored")


def test_quantized_delta_matches_a_real_quantize():
    plan = text_plan(precision="int8")
    model = _block()
    before = sum(t.nbytes for t in model.state_dict().values())
    delta = quantized_delta(model, plan, torch.float32)
    report = quantize_text_linears(model, "int8", 0, torch.float32)
    assert delta == report["quant_bytes"] - report["dense_bytes"] != 0
    after = sum(t.nbytes for t in model.state_dict().values())
    assert after - before == delta


def test_copy_bytes():
    assert [pow2(n) for n in (1, 2, 3, 4096, 4097)] == [1, 2, 4, 4096, 8192]
    fill = _Fill.__new__(_Fill)
    view = torch.empty(3, 4, dtype=torch.bfloat16)
    private = torch.empty(5, dtype=torch.int8)
    fill.views, fill.dtypes = {id(view)}, {id(view): torch.float32}
    assert copy_bytes([view, private], fill, pinned=False) == 48
    assert copy_bytes([view, private], fill, pinned=True) == 64 + (8 - 5)
