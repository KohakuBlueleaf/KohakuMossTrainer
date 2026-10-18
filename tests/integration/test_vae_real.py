"""VAE placements on the real FLUX.2 VAE: the same latents under resident / offload /
ring (eager encode in each; `compile="auto"` compiles the resident one only), no VAE
weight on the device between passes for offload / ring, and the
latent encode-ahead runs one pass per `ahead` batches."""

import numpy as np
import pytest
import torch

from kohakumoss.encoders.vae import FluxVae
from kohakumoss.offload.encoders.vae import place_vae
from kohakumoss.training.vae import LatentAhead
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.cuda, pytest.mark.slow]
SIDE = 256


def _images(n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (SIDE, SIDE, 3), dtype=np.uint8) for _ in range(n)]


def _hw(n: int) -> torch.Tensor:
    return torch.full((n, 2), SIDE // 16, dtype=torch.long)


@pytest.fixture(scope="module")
def vae_path(weights_root):
    return weights_root / "base" / "vae"


def _placed(vae_path, memory: str, **spec):
    vae = FluxVae.from_pretrained(vae_path, "cpu", torch.bfloat16)
    return place_vae(vae, {"memory": memory, **spec}, torch.device("cuda"))


def _on_device(placed) -> bool:
    return any(p.device.type != "cpu" for p in placed.vae.vae.parameters())


def _equal(a, b) -> None:
    assert torch.equal(a, b)


def test_latents_identical_across_placements(vae_path):
    images, hw = _images(2, 1), _hw(2)
    out = {
        m: _placed(vae_path, m, compile="off").encode_images(images, hw).cpu()
        for m in ("resident", "offload", "ring")
    }
    _equal(out["offload"], out["resident"])
    _equal(out["ring"], out["resident"])
    other = _placed(vae_path, "offload", compile="off")
    other = other.encode_images(_images(2, 2), hw).cpu()
    must_fail(lambda: _equal(other, out["resident"]), "different images")


@pytest.mark.parametrize("memory", ("offload", "ring"))
def test_no_vae_weight_on_device_between_passes(vae_path, memory):
    placed = _placed(vae_path, memory)
    placed.encode_images(_images(2, 1), _hw(2))
    assert not _on_device(placed)
    assert placed.stats["h2d_bytes"] > 0
    resident = _placed(vae_path, "resident")
    must_fail(lambda: _assert_released(resident), "a placement that keeps its weights")


def _assert_released(placed) -> None:
    assert not _on_device(placed)


class _Batch:
    def __init__(self, i):
        self.latents = None
        self.pixels = _images(2, seed=i)
        self.hw = _hw(2)


class _Loader:
    def __init__(self, n):
        self.batches = [_Batch(i) for i in range(n)]
        self.start = 0

    def __iter__(self):
        for hb in self.batches[self.start :]:
            self.start += 1
            yield hb

    def state_dict(self):
        return {"start": self.start}

    def load_state_dict(self, s):
        self.start = s["start"]


def test_encode_ahead_runs_one_pass_per_ahead_batches(vae_path):
    placed = _placed(vae_path, "offload", ahead=4)
    ahead = LatentAhead(_Loader(8), placed, 4)
    for _, hb in zip(range(4), ahead):
        ahead.take(hb.pixels, hb.hw)
    assert placed.stats["passes"] == 1 and placed.stats["images"] == 8
    assert ahead.state_dict()["start"] == 4
    per_batch = _placed(vae_path, "offload")
    plain = LatentAhead(_Loader(8), per_batch, 1)
    for _, hb in zip(range(4), plain):
        plain.take(hb.pixels, hb.hw)
    must_fail(lambda: _assert_passes(per_batch, 1), "one VAE pass per batch")


def _assert_passes(placed, n) -> None:
    assert placed.stats["passes"] == n
