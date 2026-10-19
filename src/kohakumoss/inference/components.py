"""The pipeline's text encoder and VAE: the text encoder places itself by its
TextEncoderPlan (`free_after_use` drops it after its first pass), the VAE by its VaePlan
(kohakumoss.offload.encoders.vae.PlacedVae: resident, offloaded or streamed per decode).
"""

from collections.abc import Sequence
from functools import partial

import torch
from torch import Tensor

from kohakumoss.offload.encoders.vae import PlacedVae, place_vae


def _same(items):
    return items


def _no_text_encoder(*args):
    raise RuntimeError("no text encoder held; pass prompt / negative embeddings")


def _encode_with(text_encoder, prompts: Sequence[str]) -> list[Tensor]:
    return text_encoder.encode(list(prompts))


class Components:
    """`encode` / `empty` through the text encoder (None: embeddings only), `decode`
    through the VAE; a bare FluxVae is placed resident on `device`."""

    def __init__(self, text_encoder, vae, device):
        self.text_encoder = text_encoder
        self.device = torch.device(device)
        if vae is not None and not isinstance(vae, PlacedVae):
            vae = place_vae(vae, {"memory": "resident"}, self.device)
        self.vae = vae
        if text_encoder is None:
            self.encode = self.empty = _no_text_encoder
        else:
            self.encode = partial(_encode_with, text_encoder)
            self.empty = text_encoder.empty

    @property
    def vae_on_device(self) -> bool:
        """The VAE's weights stay on the device between decodes."""
        return self.vae is not None and self.vae.placement.name == "resident"

    def decode(self, latents: list[Tensor], wrap=_same) -> list[Tensor]:
        """Decoded images, one VAE pass each; `wrap` wraps the latent iteration (e.g. a
        progress bar)."""
        return [self.vae.decode(z) for z in wrap(latents)]
