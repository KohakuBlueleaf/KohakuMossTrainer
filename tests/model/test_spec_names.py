"""Encoder spec naming is total over torch dtypes: the fp64 precision-reference text
encoder gets a spec (its dtype's torch name), the training dtypes keep their short
names."""

import pytest
import torch

from kohakumoss.encoders.specs import text_spec, vae_spec
from kohakumoss.types.plans import TextEncoderPlan
from kohakumoss_kernels.dtypes import NAME_OF
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _names_the_reference(spec_of) -> None:
    try:
        spec = spec_of(torch.float64)
    except KeyError as err:
        raise AssertionError(f"no spec name for float64: {err}") from err
    assert spec == "float64/compute", spec


def test_every_dtype_has_a_spec_name():
    plan = TextEncoderPlan()
    _names_the_reference(lambda dtype: text_spec(plan, dtype))
    assert text_spec(plan, torch.bfloat16) == "bf16/compute"
    assert vae_spec(torch.float32) == "fp32" and vae_spec(torch.float16) == "fp16"
    must_fail(
        lambda: _names_the_reference(lambda dtype: f"{NAME_OF[dtype]}/compute"),
        "the short-name table alone",
    )
