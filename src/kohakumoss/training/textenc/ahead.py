"""Encode-ahead for the online text cache: one text-encoder pass covers the cache
misses of the next `n` host batches, so streamed / offloaded weights cross the bus once
per `n` batches. The wrapper is the loader the driver and checkpoints see: its state
is the position of the first batch not yet handed out."""

from collections import deque

from torch import Tensor


class TextAhead:
    """Iterates `loader`; every `n` batches it encodes the prompts of the batches that
    miss the text cache in one `encode` call and keeps the results until `take(hb)`."""

    def __init__(self, loader, encode, n: int):
        self.loader = loader
        self.dataset = getattr(loader, "dataset", None)
        self.encode = encode
        self.n = max(int(n), 1)
        self.buf: deque = deque()
        self.ready: dict[int, list[Tensor]] = {}

    def _fill(self, it) -> None:
        batches = [next(it) for _ in range(self.n)]
        misses = [hb for hb in batches if hb.text is None and hb.prompts]
        prompts = [p for hb in misses for p in hb.prompts]
        encoded = self.encode(prompts) if prompts else []
        at = 0
        for hb in misses:
            self.ready[id(hb)] = encoded[at : at + len(hb.prompts)]
            at += len(hb.prompts)
        self.buf.extend(batches)

    def __iter__(self):
        self.buf.clear()
        self.ready.clear()
        it = iter(self.loader)
        while True:
            if not self.buf:
                self._fill(it)
            yield self.buf.popleft()

    def take(self, hb) -> list[Tensor]:
        """The encodings of `hb.prompts` (from the last pass, else encoded now)."""
        parts = self.ready.pop(id(hb), None)
        return self.encode(hb.prompts) if parts is None else parts

    def state_dict(self) -> dict:
        state = dict(self.loader.state_dict())
        state["start"] = state["start"] - len(self.buf)
        return state

    def load_state_dict(self, state: dict) -> None:
        self.buf.clear()
        self.ready.clear()
        self.loader.load_state_dict(state)
