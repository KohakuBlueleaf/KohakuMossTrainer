"""The device speed measurement of the stage split leaves the caller's random stream
alone."""

import pytest
import torch

from kohakumoss.parallel.pipeline.balance import _matmul_tflops, measure_speed
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def check_rng_untouched(measure) -> None:
    """A speed measurement leaves the caller's CPU random stream where it was."""
    torch.manual_seed(123)
    want = torch.rand(4)
    torch.manual_seed(123)
    assert measure() > 0
    assert torch.equal(torch.rand(4), want), "the measurement moved the generator"


def test_measure_speed_keeps_the_rng():
    cpu = torch.device("cpu")
    check_rng_untouched(lambda: measure_speed(cpu, 64, 1))
    must_fail(
        lambda: check_rng_untouched(lambda: _matmul_tflops(cpu, 64, 1)),
        "draws on the caller's generator",
    )
