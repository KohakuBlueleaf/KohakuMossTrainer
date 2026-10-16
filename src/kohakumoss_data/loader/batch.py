"""MossHostBatch: HostBatch plus the plan coordinates online caches need to write back."""

from dataclasses import dataclass, field

from torch import Tensor

from kohakumoss_data.types import HostBatch


@dataclass
class MossHostBatch(HostBatch):
    """`index` / `rank` locate the plan; `latent_keys` / `text_keys` are the cache keys of
    each sample (empty when that cache is off). `text_limit[i]` > 0 caps sample i's
    text tokens (truncate_text). When `text` is set it is already capped and `text_len`
    is final; with `prompts` the consumer encodes, keeps the first `text_limit` rows
    and fills `text_len`. When `pixels` are set the consumer encodes the full grid and
    gathers `img_keep` itself; cached `latents` are already gathered."""

    index: int = -1
    rank: int = 0
    latent_keys: list[str] = field(default_factory=list)
    text_keys: list[str] = field(default_factory=list)
    text_limit: list[int] = field(default_factory=list)

    def pin_memory(self) -> "MossHostBatch":
        """DataLoader pin-memory hook."""
        return self.pin()

    def tensors(self) -> dict[str, Tensor]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, Tensor)}
