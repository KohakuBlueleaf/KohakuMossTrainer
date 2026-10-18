"""The real VAE's latent statistics: FluxVae keeps the file's BatchNorm mean and
sqrt(var + eps) bit for bit in fp32, whatever dtype the VAE runs in (bf16, fp16)."""

import json

import pytest
import torch
from safetensors import safe_open

from kohakumoss.encoders.vae import FluxVae
from tests._util.negative import must_fail

pytestmark = [pytest.mark.real_weights, pytest.mark.slow]
MEAN, VAR = "bn.running_mean", "bn.running_var"


@pytest.fixture(scope="module")
def file_stats(weights_root) -> tuple:
    """(running mean, running var, batch_norm_eps) as stored in the VAE file."""
    vae_dir = weights_root / "base" / "vae"
    with safe_open(str(vae_dir / "diffusion_pytorch_model.safetensors"), "pt") as f:
        mean, var = f.get_tensor(MEAN), f.get_tensor(VAR)
    eps = json.loads((vae_dir / "config.json").read_text(encoding="utf-8"))
    return mean, var, eps["batch_norm_eps"]


def check_stats(vae: FluxVae, mean, var, eps: float) -> None:
    assert vae.mean.dtype == vae.std.dtype == torch.float32
    assert torch.equal(vae.mean, mean.float())
    assert torch.equal(vae.std, (var.float() + eps).sqrt())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_vae_keeps_the_file_statistics(dtype, weights_root, file_stats):
    mean, var, eps = file_stats
    assert mean.dtype == var.dtype == torch.float32, (mean.dtype, var.dtype)
    vae = FluxVae.from_pretrained(weights_root / "base", "cpu", dtype)
    assert vae.dtype == dtype
    check_stats(vae, mean, var, eps)
    rounded = (mean.to(dtype).float(), var.to(dtype).float())
    must_fail(lambda: check_stats(vae, *rounded, eps), f"statistics rounded to {dtype}")
