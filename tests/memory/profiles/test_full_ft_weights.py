"""A full finetune's device parameters are its masters (fp32, the compute dtype under
adamw16): the estimate counts them once at that width, and the planner's text-pass peak
sits on every part allocated for the run (EMA, state and gradient rings included)
(model-free)."""

import importlib

import pytest

from kohakumoss.memory import planner
from kohakumoss.memory.profiles import resolve
from kohakumoss.offload.encoders.text_memory import TextSizes
from kohakumoss.types.arch import PRESETS
from tests._util.negative import must_fail

E = importlib.import_module("kohakumoss.memory.estimate")
pytestmark = pytest.mark.cpu
ARCH = PRESETS["base"]
N = E.total_params(ARCH)


def _work(**kw):
    return E.Workload(ARCH, img_tokens=4096, txt_tokens=300, full_ft=True, **kw)


def check_master_width(estimate) -> None:
    gpu = estimate(_work(), resolve("speed", device="cpu", trainable_base=True))
    assert "master" not in gpu.parts
    assert gpu.parts["weights"] + gpu.parts["optim"] == N * 12, gpu.gib()
    stream = resolve("ckpt-resident-optstream", device="cpu", trainable_base=True)
    assert estimate(_work(), stream).parts["weights"] == N * 4
    half = estimate(_work(master_bytes=2), stream).parts["weights"]
    assert half == N * 2


def _compute_width(work, plan):
    """The estimate with full-FT parameters at the compute dtype plus a GPU master."""
    est = E.estimate(work, plan)
    est.parts["weights"] = (
        est.parts["weights"] * work.compute_bytes // work.master_bytes
    )
    if plan.optim_state == "gpu":
        est.parts["master"] = N * 4
    return est


def check_grad_width(estimate) -> None:
    resident = resolve("ckpt-resident", device="cpu", trainable_base=True)
    assert estimate(_work(), resident).parts["grads"] == N * 4
    assert estimate(_work(master_bytes=2), resident).parts["grads"] == N * 2
    swap = resolve("minimal", device="cpu", trainable_base=True)
    assert estimate(_work(), swap).parts["grads"] == E.block_params(ARCH) * 2


def _compute_grads(work, plan):
    """The estimate with every full-FT gradient at the compute dtype."""
    est = E.estimate(work, plan)
    if plan.stream != "swap":
        grads = est.parts["grads"] // work.master_bytes
        est.parts["grads"] = grads * work.compute_bytes
    return est


def test_full_ft_gradients_match_their_parameters():
    check_grad_width(E.estimate)
    must_fail(lambda: check_grad_width(_compute_grads), "fp32 gradients at 2 B")


def test_full_ft_parameters_are_the_masters():
    check_master_width(E.estimate)
    must_fail(lambda: check_master_width(_compute_width), "compute-dtype parameters")


SIZES = TextSizes(18 << 30, 2 << 30, 15 << 30, 1 << 30, 4096, 12288, 32, 128, 128, 2)


def check_pass_peak() -> None:
    plan = resolve("minimal", device="cpu", trainable_base=True)
    est = E.estimate(_work(ema_decays=1), plan)
    rings = sum(est.parts.get(p, 0) for p in ("state_ring", "grad_ring", "ema"))
    assert rings > 0
    peak, memory = planner.with_text(est, SIZES, 1 << 50, (4096, 4096), "offload")
    persistent = est.device(planner.PERSISTENT)
    assert {"state_ring", "grad_ring", "ema"} <= set(planner.PERSISTENT)
    assert peak >= persistent, (peak, persistent)


def test_text_pass_peak_counts_every_persistent_part(monkeypatch):
    check_pass_peak()
    old = ("context", "weights", "adapter", "master", "optim", "optim_slots")
    monkeypatch.setattr(planner, "PERSISTENT", old)
    must_fail(check_pass_peak, "rings and EMA left out of the persistent parts")
