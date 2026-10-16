"""Stand-ins for the data side: a deterministic stateful loader of packed HostBatches."""

import torch

from kohakumoss_data.types import HostBatch


def host_batch(
    index: int,
    seed: int,
    latent_ch: int,
    text_dim: int,
    hw=((2, 3), (3, 2)),
    text=(5, 4),
) -> HostBatch:
    g = torch.Generator().manual_seed(seed * 100003 + index)
    hw_t = torch.tensor(hw, dtype=torch.int32)
    n_img = int((hw_t[:, 0] * hw_t[:, 1]).sum())
    n_txt = int(sum(text))
    return HostBatch(
        ids=torch.arange(len(hw)) + index * len(hw),
        hw=hw_t,
        text_len=torch.tensor(text, dtype=torch.int32),
        text_drop=torch.zeros(len(hw), dtype=torch.bool),
        weight=torch.ones(len(hw)),
        latents=torch.randn(n_img, latent_ch, generator=g),
        text=torch.randn(n_txt, text_dim, generator=g),
    )


class FakeLoader:
    """Infinite, deterministic; state `{seed, start}` resumes exactly."""

    def __init__(self, latent_ch: int, text_dim: int, seed: int = 0):
        self.latent_ch = latent_ch
        self.text_dim = text_dim
        self.seed = seed
        self.start = 0

    def __iter__(self):
        while True:
            hb = host_batch(self.start, self.seed, self.latent_ch, self.text_dim)
            self.start += 1
            yield hb

    def state_dict(self) -> dict:
        return {"seed": self.seed, "start": self.start}

    def load_state_dict(self, state: dict) -> None:
        self.seed = state["seed"]
        self.start = state["start"]
