"""Encoder compute specs, model-free: what tells two encodings apart (VAE dtype; text
dtype, precision, quantization group, fp8 recipe) is in the spec, what does not
(placement, batching, layout) is not, and the cache encoder carries its spec."""

import pytest
import torch

from kohakumoss.encoders.specs import SpecEncoder, cache_specs, text_spec, vae_spec
from kohakumoss.encoders.text.plan import text_plan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
BF16, FP16 = torch.bfloat16, torch.float16


def spec(dtype=BF16, **fields) -> str:
    return text_spec(text_plan(fields), dtype)


def _distinct(specs: list[str]) -> None:
    assert len(set(specs)) == len(specs), specs


def test_text_spec_tells_computations_apart():
    variants = [
        spec(),
        spec(FP16),
        spec(precision="int8"),
        spec(precision="int8", quant_group=64),
        spec(precision="fp8"),
        spec(precision="fp8_compute", fp8_recipe="row"),
        spec(precision="fp8_compute", fp8_recipe="block"),
        spec(precision="mxfp8"),
    ]
    _distinct(variants)
    assert spec() == "bf16/compute" and spec(precision="mxfp8") == "bf16/mxfp8/mx"
    same = [spec(memory="stream"), spec(memory="offload", batch=4, layout="padded")]
    assert set(same) == {spec()}

    def precision_only(plan, dtype) -> str:
        return plan.precision

    must_fail(
        lambda: _distinct([precision_only(text_plan({}), d) for d in (BF16, FP16)]),
        "a spec without the dtype",
    )


def test_vae_spec_and_the_knobs_it_fills():
    assert [vae_spec(d) for d in (torch.float32, BF16, FP16)] == [
        "fp32",
        "bf16",
        "fp16",
    ]
    knobs = cache_specs(FP16, text_plan({"precision": "int8"}), FP16)
    assert knobs == {"LATENT_ENCODER_SPEC": "fp16", "TEXT_ENCODER_SPEC": "fp16/int8/g0"}


def test_the_cache_encoder_carries_its_spec():
    encode = SpecEncoder(lambda images, hw: (images, hw), "bf16")
    assert encode.encoder_spec == "bf16" and encode([1], 2) == ([1], 2)
