"""The step-memory estimate against a recorded real run (model-free): the MOSS base DiT,
LoKr factor 8 full-matrix on the attn-mlp linears, fp16, the `speed` profile, one
micro-batch of 8124 image + 188 text tokens. Recorded at step 1 (RTX PRO 6000):
53.23 GiB peak allocated, 55.80 GiB on the device. The estimate's allocator parts and
its device total must each fall within BAND of those."""

import pytest

from kohakumoss.memory.estimate import GIB, Workload, estimate
from kohakumoss.memory.profiles import resolve
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
ARCH = PRESETS["base"]
IMG_TOKENS, TXT_TOKENS = 8124, 188
PEAK_ALLOCATED_GIB = 53.23
DEVICE_GIB = 55.80
BAND = (0.85, 1.10)
FACTOR = 8


def lokr_params(arch, factor: int = FACTOR) -> int:
    """LoKr full-matrix parameters on every block's q, k, v, o and FFN linears: a
    factor x factor w1 and an (out / factor) x (in / factor) w2 per linear."""
    d, f = arch.dim, arch.ffn_dim
    shapes = [(d, d)] * 4 + [(f, d), (f, d), (d, f)]
    per_block = sum(factor * factor + (o // factor) * (i // factor) for o, i in shapes)
    return arch.depth * per_block


def _work(img: int = IMG_TOKENS, txt: int = TXT_TOKENS) -> Workload:
    return Workload(
        ARCH, img_tokens=img, txt_tokens=txt, trainable_params=lokr_params(ARCH)
    )


def check_recorded(work: Workload, profile: str) -> None:
    est = estimate(work, resolve(profile, device="cpu", trainable_base=False))
    allocated = sum(v for k, v in est.parts.items() if k != "context") / GIB
    device = est.total / GIB
    lo, hi = BAND
    assert lo <= allocated / PEAK_ALLOCATED_GIB <= hi, (allocated, est.gib())
    assert lo <= device / DEVICE_GIB <= hi, (device, est.gib())


def test_speed_estimate_matches_the_recorded_peak():
    check_recorded(_work(), "speed")
    must_fail(lambda: check_recorded(_work(), "balanced"), "a checkpointed estimate")
    half = _work(IMG_TOKENS // 2, TXT_TOKENS // 2)
    must_fail(lambda: check_recorded(half, "speed"), "half the recorded tokens")
