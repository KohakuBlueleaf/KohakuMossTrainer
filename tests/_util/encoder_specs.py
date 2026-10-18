"""Encoder-spec test helpers: the latent cache key spec of a VAE encoder's arguments."""

from pathlib import Path

import torch

from kohakumoss.encoders.specs import vae_spec
from kohakumoss.encoders.vae import resolve_vae_dtype


def vae_encoder_spec(
    path: str | Path = "", device: torch.device | str = "cuda", dtype="auto", **_
) -> str:
    """The `encoder_spec` `vae_encoder(...)` gives for the same arguments, without
    loading the weights."""
    return vae_spec(resolve_vae_dtype(dtype, device))
