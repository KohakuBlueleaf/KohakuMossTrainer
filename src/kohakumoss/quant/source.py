"""Quantization source: a weight's checkpoint values, quantized instead of its live
copy when that copy is exactly their cast (times a fold multiplier) to another dtype,
and in place of a copy the load left unread. See docs/internals/quant-kernels.md#source.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from kohakumoss_common.log import get_logger
from kohakumoss_kernels.dtypes import SAFETENSORS as SAFETENSORS_DTYPES

STORED, SAME, NO_SOURCE, NOT_THE_CAST = "stored", "same", "no_source", "not_the_cast"
DEFERRED = "deferred"
SOURCES = {
    STORED: "from the stored values",
    DEFERRED: "from the stored values (no 16-bit copy loaded)",
    SAME: "from the live weight (stored in its dtype)",
    NO_SOURCE: "from the live weight (no stored values)",
    NOT_THE_CAST: (
        "from the live weight (not the stored values' cast: a merged delta or "
        "another checkpoint)"
    ),
}
log = get_logger(__name__)


class SafetensorsFiles:
    """Tensors of safetensors files, opened lazily: keys and dtypes from the headers,
    one tensor per `get`."""

    def __init__(self, files: list[Path]):
        self._where = {}
        for f in files:
            handle = safe_open(str(f), framework="pt", device="cpu")
            for key in handle.keys():
                self._where[key] = handle

    def keys(self) -> list[str]:
        return list(self._where)

    def dtype(self, key: str) -> torch.dtype | None:
        return SAFETENSORS_DTYPES.get(self._where[key].get_slice(key).get_dtype())

    def get(self, key: str) -> Tensor:
        return self._where[key].get_tensor(key)


class SourceWeights:
    """Stored weights by module weight name ("<path>.weight"): `resolve(name)` gives the
    key in `tensors` (keys / dtype / get) or None; `dtype(name)` reads the header only.
    """

    def __init__(self, tensors, resolve: Callable[[str], str | None]):
        self.tensors = tensors
        self.resolve = resolve

    def dtype(self, name: str) -> torch.dtype | None:
        key = self.resolve(name)
        return None if key is None else self.tensors.dtype(key)

    def get(self, name: str) -> Tensor:
        return self.tensors.get(self.resolve(name))


def suffix_resolver(keys: list[str]) -> Callable[[str], str | None]:
    """`name` -> the one key equal to it or ending in "." + name (None: absent or
    ambiguous)."""
    by_suffix: dict[str, list[str]] = {}
    for key in keys:
        parts = key.split(".")
        for i in range(len(parts)):
            by_suffix.setdefault(".".join(parts[i:]), []).append(key)

    def resolve(name: str) -> str | None:
        hits = by_suffix.get(name, [])
        return hits[0] if len(hits) == 1 else None

    return resolve


def text_source(model_dir: str | Path) -> SourceWeights | None:
    """The stored weights of a text-encoder checkpoint dir, by text-model weight name
    (None when the dir holds no safetensors)."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        return None
    tensors = SafetensorsFiles(files)
    return SourceWeights(tensors, suffix_resolver(tensors.keys()))


def _rows(scale: Tensor | float, like: Tensor) -> Tensor | float:
    if isinstance(scale, Tensor):
        return scale.to(like.device, torch.float32).reshape(
            -1, *([1] * (like.dim() - 1))
        )
    return float(scale)


def source_weight(
    live: Tensor,
    source: SourceWeights | None,
    name: str,
    scale: Tensor | float | None = None,
    work: torch.device | str | None = None,
) -> tuple[Tensor, str]:
    """(values to quantize for the live weight `name` on `work` (default: where `live`
    is), where they come from: one of SOURCES): its stored values times the fold
    multiplier `scale` (fp32) when they round to `live` exactly and are stored in
    another dtype or folded since ("stored"), else `live` itself, with the reason."""
    dev = torch.device(work) if work is not None else live.device
    live = live.detach()
    if source is None or live.device.type == "meta":
        return live.to(dev), NO_SOURCE
    stored_dtype = source.dtype(name)
    if stored_dtype is None:
        return live.to(dev), NO_SOURCE
    if stored_dtype == live.dtype and scale is None:
        return live.to(dev), SAME
    stored = source.get(name)
    if tuple(stored.shape) != tuple(live.shape):
        return live.to(dev), NOT_THE_CAST
    values = stored.to(dev, torch.float32)
    if scale is not None:
        values = values * _rows(scale, values)
    here = live.to(dev)
    if torch.equal(values.to(live.dtype), here):
        return values, STORED
    return here, NOT_THE_CAST


def deferred_weight(
    source: SourceWeights | None,
    name: str,
    dtype: torch.dtype,
    scale: Tensor | float | None = None,
    work: torch.device | str = "cpu",
) -> Tensor:
    """The values to quantize for weight `name`, which the load left unread: its stored
    values on `work`, in `dtype` (the compute dtype) when stored in it and unscaled
    (what a loaded copy would hold), else in fp32 times the fold multiplier `scale`;
    ValueError when `source` does not hold them."""
    if source is None or source.dtype(name) is None:
        raise ValueError(
            f"{name} was left unread at load and no checkpoint source holds it"
        )
    stored = source.get(name)
    if stored.dtype == dtype and scale is None:
        return stored.to(work)
    values = stored.to(work, torch.float32)
    return values * _rows(scale, values) if scale is not None else values


def tally(counts: dict[str, int], kind: str) -> None:
    counts[kind] = counts.get(kind, 0) + 1


def log_sources(what: str, counts: dict[str, int]) -> None:
    """One INFO line: how many linears of `what` were quantized from the stored values
    and from the live weight, by reason."""
    if not counts:
        return
    parts = ", ".join(f"{SOURCES[k]} {n}" for k, n in sorted(counts.items()))
    log.info(f"{what}: quantized {sum(counts.values())} linears: {parts}")
