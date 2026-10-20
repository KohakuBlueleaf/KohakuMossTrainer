"""VAE placement, model-free: the auto choice, the training knob check (every placement
with every latent cache mode) and each
placement class's name (the text encoder's too). The
placements themselves (same latents, no weight on the device between passes, encode
ahead) run on the real VAE in tests/integration/test_vae_real.py."""

import pytest

from kohakumoss.offload.encoders.text import PLACEMENTS as TEXT_PLACEMENTS
from kohakumoss.offload.encoders.vae import PLACEMENTS as VAE_PLACEMENTS
from kohakumoss.offload.encoders.vae import choose_vae_memory
from kohakumoss.training.checks.encoders import vae_problems
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def test_auto_picks_the_fastest_placement_that_fits():
    gib = 1 << 30
    assert choose_vae_memory(gib, 24 * gib, 8 * gib, 1 << 20, False) == "resident"
    assert choose_vae_memory(gib, 10 * gib, 9 * gib, 1 << 20, False) == "offload"
    assert choose_vae_memory(gib, 3 * gib, 6 * gib, 1 << 20, False) == "ring"
    big = 4096 * 4096
    assert choose_vae_memory(gib, 12 * gib, 4 * gib, big, True) == "resident"
    must_fail(
        lambda: _assert_choice(gib, 10 * gib, 9 * gib, "resident"),
        "resident chosen without room next to the DiT step",
    )


def _assert_choice(weights, budget, reserve, want):
    assert choose_vae_memory(weights, budget, reserve, 1 << 20, False) == want


def test_vae_knob_problems():
    for memory in ("offload", "ring", "cpu", "resident", "auto"):
        for cache in ("offline", "online", "off"):
            _assert_clean({"memory": memory}, cache)
    assert vae_problems({"VAE_ENCODE": {"memroy": "auto"}, "LATENT_CACHE": "off"})
    must_fail(lambda: _assert_clean({"memroy": "cpu"}, "off"), "an unknown field")


def _assert_clean(spec, cache):
    assert vae_problems({"VAE_ENCODE": spec, "LATENT_CACHE": cache}) == []


def assert_named(placements) -> None:
    """Every placement class reports the memory value it implements (the reports, the
    GPU-peak estimate and `vae_on_device` read the name)."""
    assert {k: cls.name for k, cls in placements.items()} == {k: k for k in placements}


def test_every_placement_names_its_memory():
    assert_named(VAE_PLACEMENTS)
    assert_named(TEXT_PLACEMENTS)
    broken = {**VAE_PLACEMENTS, "cpu": VAE_PLACEMENTS["resident"]}
    must_fail(lambda: assert_named(broken), "the host VAE under the resident class")
