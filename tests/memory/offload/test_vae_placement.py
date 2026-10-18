"""VAE placement, model-free: the auto choice and the training knob check. The
placements themselves (same latents, no weight on the device between passes, encode
ahead) run on the real VAE in tests/integration/test_vae_real.py."""

import pytest

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
    assert (
        vae_problems({"VAE_ENCODE": {"memory": "offload"}, "LATENT_CACHE": "online"})
        == []
    )
    assert (
        vae_problems({"VAE_ENCODE": {"memory": "cpu"}, "LATENT_CACHE": "offline"}) == []
    )
    bad = vae_problems({"VAE_ENCODE": {"memory": "cpu"}, "LATENT_CACHE": "online"})
    assert bad and "offline cache pass" in bad[0]
    assert vae_problems({"VAE_ENCODE": {"memroy": "auto"}, "LATENT_CACHE": "off"})
    must_fail(
        lambda: _assert_clean({"memory": "cpu"}, "off"), "CPU VAE with online encoding"
    )


def _assert_clean(spec, cache):
    assert vae_problems({"VAE_ENCODE": spec, "LATENT_CACHE": cache}) == []
