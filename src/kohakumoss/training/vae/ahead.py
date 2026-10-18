"""Encode-ahead for the online latent cache: one VAE pass covers the cache misses of the
next `n` host batches, so an offloaded / streamed VAE crosses the bus once per `n`
batches and is released before the DiT step. The wrapper is the loader the driver and
checkpoints see: its state is the position of the first batch not yet handed out."""

from collections import deque

from torch import Tensor


class LatentAhead:
    """Iterates `loader`; every `n` batches it encodes the pixels of the batches that
    miss the latent cache in one `encode_batches` pass and keeps the latents until
    `take(pixels, hw)`; `batch_of(hb)` maps a batch to its stored latents."""

    def __init__(self, loader, vae, n: int):
        self.loader = loader
        self.dataset = getattr(loader, "dataset", None)
        self.vae = vae
        self.n = max(int(n), 1)
        self.buf: deque = deque()
        self.ready: dict[int, Tensor] = {}

    def _fill(self, it) -> None:
        batches = [next(it) for _ in range(self.n)]
        misses = [hb for hb in batches if hb.latents is None and hb.pixels is not None]
        if misses:
            encoded = self.vae.encode_batches([(hb.pixels, hb.hw) for hb in misses])
            for hb, z in zip(misses, encoded):
                self.ready[id(hb.pixels)] = z
        self.buf.extend(batches)

    def __iter__(self):
        self.buf.clear()
        self.ready.clear()
        it = iter(self.loader)
        while True:
            if not self.buf:
                self._fill(it)
            yield self.buf.popleft()

    def take(self, pixels, hw) -> Tensor:
        """The latents of `pixels` (from the last pass, else encoded now)."""
        z = self.ready.pop(id(pixels), None)
        return self.vae.encode_images(pixels, hw) if z is None else z

    def state_dict(self) -> dict:
        state = dict(self.loader.state_dict())
        state["start"] = state["start"] - len(self.buf)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.buf.clear()
        self.ready.clear()
        self.loader.load_state_dict(state)
