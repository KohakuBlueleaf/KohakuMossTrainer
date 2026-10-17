"""The applier opts LyCORIS into the measured tier, KohakuMoss's compile mode and the
half-matmul policy of the model's compute dtype.

LyCORIS's library defaults are its released behaviour (the first capable tier,
compile `auto`, fp32 matmuls on fp32 operands); KohakuMoss's applier must switch them
at bind time.
"""

import lycoris.kernels as lyk
import pytest
import torch
from _adcases import make_spec
from _adtoy import toy

from kohakumoss.adapters import apply, load_adapters
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


@pytest.fixture
def library_defaults():
    lyk.set_tier_mode(None)
    lyk.set_compile_dynamic(None)
    lyk.set_half_matmul(None)
    yield
    lyk.set_tier_mode(None)
    lyk.set_compile_dynamic(None)
    lyk.set_half_matmul(None)


def _check_opted_in(spec, compute):
    assert lyk.tier_mode() == "measured", "applier left LyCORIS on the first tier"
    assert lyk.compile_dynamic() == spec.compile_dynamic
    assert lyk.half_matmul() == compute, "LyCORIS matmuls left off the 16-bit policy"


def test_apply_opts_in(library_defaults, monkeypatch):
    monkeypatch.delenv("LYCORIS_KERNEL_TIER", raising=False)
    monkeypatch.delenv("LYCORIS_KERNEL_COMPILE_DYNAMIC", raising=False)
    monkeypatch.delenv("LYCORIS_KERNEL_HALF_MATMUL", raising=False)
    spec = make_spec("lora", mode="bypass", backend="torch")
    assert spec.compile_dynamic == "tokens"
    apply(toy(dtype=torch.bfloat16), spec)
    _check_opted_in(spec, torch.bfloat16)


def test_an_fp32_model_keeps_fp32_matmuls(library_defaults):
    spec = make_spec("lora", backend="torch")
    apply(toy(), spec)
    _check_opted_in(spec, None)


def test_a_conditioning_layer_sets_the_policy(library_defaults):
    model = toy()
    model.get_submodule("shared_block_modulation").set_input_dtype(torch.float16)
    spec = make_spec("lora", backend="torch")
    apply(model, spec)
    _check_opted_in(spec, torch.float16)


def test_inference_loading_sets_the_policy(library_defaults, tmp_path):
    source = toy()
    aset = apply(source, make_spec("lora", backend="torch"))
    path = tmp_path / "a.safetensors"
    aset.export(str(path), "comfy", dtype=None)
    lyk.set_half_matmul(None)
    load_adapters(toy(dtype=torch.bfloat16), [(str(path), 1.0)], "bypass")
    assert lyk.half_matmul() == torch.bfloat16


def test_check_rejects_library_defaults(library_defaults):
    spec = make_spec("lora", mode="bypass", backend="torch")
    must_fail(lambda: _check_opted_in(spec, torch.bfloat16), "LyCORIS library defaults")


def test_check_rejects_a_missing_half_policy(library_defaults):
    spec = make_spec("lora", mode="bypass", backend="torch")
    apply(toy(dtype=torch.bfloat16), spec)
    lyk.set_half_matmul(None)
    must_fail(lambda: _check_opted_in(spec, torch.bfloat16), "half matmul off")
