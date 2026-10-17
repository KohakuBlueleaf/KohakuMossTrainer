"""Where the text encoder and VAE live between uses: all on GPU, offloaded, or TE once."""

import gc
from collections.abc import Sequence

import torch
from torch import Tensor


def _no_text_encoder(*args):
    raise RuntimeError("no text encoder held; pass prompt / negative embeddings")


class Resident:
    """Text encoder and VAE stay on the compute device."""

    def __init__(self, text_encoder, vae, device):
        self.text_encoder = text_encoder
        self.vae = vae
        self.device = torch.device(device)
        if vae is not None:
            vae.to(self.device)
        self._bind_text(text_encoder)

    def _bind_text(self, text_encoder) -> None:
        if text_encoder is None:
            self.encode = self.empty = _no_text_encoder
        else:
            self.encode = self._encode
            self.empty = text_encoder.empty

    def _encode(self, prompts: Sequence[str]) -> list[Tensor]:
        return self.text_encoder.encode(list(prompts))

    def decode(self, latents: list[Tensor]) -> list[Tensor]:
        return [self.vae.decode(z.to(self.device)) for z in latents]


class Offloaded(Resident):
    """Both encoders park on CPU and move to the device only while they run."""

    def __init__(self, text_encoder, vae, device):
        super().__init__(text_encoder, None, device)
        self.vae = vae
        if vae is not None:
            vae.to("cpu")

    def decode(self, latents: list[Tensor]) -> list[Tensor]:
        self.vae.to(self.device)
        try:
            return [self.vae.decode(z.to(self.device)) for z in latents]
        finally:
            self.vae.to("cpu")


class TextOnce(Resident):
    """The text encoder runs for the first encode call only, then is freed."""

    def _encode(self, prompts: Sequence[str]) -> list[Tensor]:
        out = self.text_encoder.encode(list(prompts))
        empty = self.text_encoder.empty()
        self.text_encoder = None
        self.encode = _no_text_encoder
        self.empty = lambda: empty
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        return out


RESIDENCY = {"gpu": Resident, "offload": Offloaded, "te_once": TextOnce}
