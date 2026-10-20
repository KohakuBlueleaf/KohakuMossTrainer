"""The sampling memory estimate per phase and the automatic VAE placement, model-free:
the text-encode, sampling and decode phases each count what the device holds then (the
peak is the largest; a VAE on the host adds nothing to decode), and "auto" tiles the
VAE pass before it rings untiled."""

from types import SimpleNamespace

import pytest
import torch

from kohakumoss.inference import peaks
from kohakumoss.model.backbone import MossBackbone
from kohakumoss.offload.encoders.vae import VaePlan, choose_vae_plan, vae_act_bytes
from kohakumoss.types.arch import PRESETS
from kohakumoss.types.plans import MemoryPlan
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
GIB = 2**30
TOKENS = 2 * (64 * 64 + 300)


def _place(te_pass: int, vae_bytes: int, resident: bool, tile: bool, where="cuda"):
    te = SimpleNamespace(footprint=lambda: {"resident_bytes": 0, "pass_bytes": te_pass})
    weights = torch.empty(vae_bytes // 2, dtype=torch.bfloat16, device="meta")
    vae = SimpleNamespace(
        vae=torch.nn.ParameterList([weights]),
        plan=VaePlan(tile=tile),
        device=torch.device(where),
    )
    return SimpleNamespace(text_encoder=te, vae=vae, vae_on_device=resident)


def _phases_add_up(est: dict, te_pass: int, vae_bytes: int, act: int) -> None:
    weights_gib = est["encode"] - round(te_pass / GIB, 2)
    assert abs(est["decode"] - (weights_gib + (vae_bytes + act) / GIB)) < 0.02, est
    assert est["peak"] == max(est["encode"], est["sampling"], est["decode"])


def test_each_phase_counts_what_it_holds():
    backbone = MossBackbone(PRESETS["base"], device="meta", dtype=torch.bfloat16)
    plan = MemoryPlan(stream="ring", stream_resident=4)
    te_pass, vae_bytes = 2 * GIB, GIB // 4
    place = _place(te_pass, vae_bytes, False, False)
    est = peaks.estimated(backbone, plan, TOKENS, place, 1024 * 1024)
    act = vae_act_bytes(1024 * 1024, False)
    _phases_add_up(est, te_pass, vae_bytes, act)
    tiled = peaks.estimated(
        backbone, plan, TOKENS, _place(te_pass, vae_bytes, False, True), 1024 * 1024
    )
    assert tiled["decode"] < est["decode"]
    must_fail(lambda: _phases_add_up(est, 0, vae_bytes, act), "no text-encoder pass")
    host = peaks.estimated(
        backbone, plan, TOKENS, _place(te_pass, vae_bytes, False, False, "cpu"), 1 << 20
    )
    _host_decode_free(host, est)
    must_fail(lambda: _host_decode_free(est, est), "a host VAE counted on the device")


def _host_decode_free(host: dict, gpu: dict) -> None:
    """A VAE on the host adds nothing to the device's decode phase."""
    assert host["decode"] < gpu["decode"] and host["vae"] == 0


def test_auto_tiles_before_it_rings_untiled():
    weights, reserve, pixels = GIB // 4, 2 * GIB, 1024 * 1024
    roomy = choose_vae_plan(VaePlan(memory="auto"), weights, 8 * GIB, reserve, pixels)
    assert roomy == VaePlan(memory="resident")
    tight = 2 * GIB + weights // 2
    plan = choose_vae_plan(VaePlan(memory="auto"), weights, tight, reserve, pixels)
    assert plan.tile and plan.memory in ("offload", "resident"), plan
    asked = choose_vae_plan(VaePlan(memory="auto", tile=True), weights, 0, 0, pixels)
    assert asked.tile and asked.memory == "ring"
