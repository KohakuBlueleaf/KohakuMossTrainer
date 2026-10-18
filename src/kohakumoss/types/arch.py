"""MOSS-Image architecture config and presets."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MossArchConfig:
    """Shape of a MOSS-Image transformer. `rope_bands * rope_axes * 2 == head_dim`."""

    dim: int = 3072
    depth: int = 42
    heads: int = 24
    ffn_dim: int = 8192
    text_dim: int = 4096
    text_blocks: int = 2
    latent_channels: int = 128
    freq_dim: int = 256
    rope_base: float = 10000.0
    rope_axes: int = 4
    eps: float = 1e-6
    time_scale: float = 1000.0
    negative_skip_layers: tuple[int, ...] = (22,)

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def rope_bands(self) -> int:
        return self.head_dim // (2 * self.rope_axes)

    def validate(self) -> None:
        problems = []
        if self.dim % self.heads:
            problems.append("dim must divide by heads")
        if self.head_dim % (2 * self.rope_axes):
            problems.append("head_dim must divide by 2 * rope_axes")
        if self.freq_dim % 2:
            problems.append("freq_dim must be even")
        if problems:
            raise ValueError("; ".join(problems))

    def override(self, **kwargs) -> "MossArchConfig":
        return replace(self, **kwargs)


PRESETS: dict[str, MossArchConfig] = {
    "base": MossArchConfig(),
    "sft": MossArchConfig(),
}
