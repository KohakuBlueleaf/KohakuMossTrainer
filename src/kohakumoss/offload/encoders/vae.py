"""VAE placement for training and sampling: resident, offloaded per pass, its blocks
through layer rings, or on the host, chosen by a VaePlan; the encode-ahead pass and
sample decodes run under the same placement. See docs/internals/offload.md#vae."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import free_bytes, streamed, with_index
from kohakumoss.offload.encoders.placement import HostCopies
from kohakumoss.offload.weights.layers import (
    LayerStreamer,
    find_layer_list,
    tensor_bytes,
    tensor_ids,
)
from kohakumoss_common.calls import nothing
from kohakumoss_kernels.hardware.current import on_device
from kohakumoss_kernels.hardware.host_gemm import host_gemm

MEMORY = ("resident", "offload", "ring", "cpu", "auto")
MEMORY_ALIASES = {"stream": "ring"}
COMPILE = ("auto", "on", "off")
ACT_BYTES_PER_PIXEL = 1024


@dataclass(frozen=True)
class VaePlan:
    """VAE_ENCODE: `memory` placement, `batch` same-size images per encode call, `ahead`
    host batches encoded per pass, `tile` tiled / sliced encode and decode, `compile`
    channels_last + torch.compile of encode / decode ('auto': on for a resident CUDA
    VAE)."""

    memory: str = "resident"
    batch: int = 1
    ahead: int = 1
    tile: bool = False
    compile: str = "auto"


def compiles(plan: VaePlan, device: torch.device) -> bool:
    """Whether `plan` runs the VAE compiled on `device`."""
    match plan.compile:
        case "on":
            return True
        case "off":
            return False
    return device.type == "cuda" and plan.memory == "resident" and not plan.tile


def vae_plan(spec: dict | VaePlan | None = None) -> VaePlan:
    """A VaePlan from a plan or a dict of its fields, with `memory` in its primary
    spelling (`MEMORY_ALIASES`: "stream" is "ring"); raises listing unknown fields."""
    if isinstance(spec, VaePlan):
        plan = spec
    else:
        values = dict(spec or {})
        known = {f.name for f in fields(VaePlan)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"unknown VAE plan fields {unknown}; valid {sorted(known)}"
            )
        plan = VaePlan(**values)
    return replace(plan, memory=MEMORY_ALIASES.get(plan.memory, plan.memory))


def plan_problems(plan: VaePlan) -> list[str]:
    """Every problem with `plan`."""
    p = []
    if plan.memory not in MEMORY:
        p.append(f"VAE memory={plan.memory!r} not in {MEMORY}")
    if plan.batch < 1 or plan.ahead < 1:
        p.append("VAE batch >= 1 and ahead >= 1")
    if plan.compile not in COMPILE:
        p.append(f"VAE compile={plan.compile!r} not in {COMPILE}")
    return p


def vae_bytes(module: nn.Module) -> int:
    """Parameter bytes of the VAE module."""
    return tensor_bytes(module.parameters())


def vae_act_bytes(pixels: int, tile: bool) -> int:
    """Activation bytes of one encode / decode pass over `pixels` (a tiled pass works on
    512 x 512 tiles)."""
    return (pixels if not tile else min(pixels, 512 * 512)) * ACT_BYTES_PER_PIXEL


def choose_vae_memory(
    weights: int, budget: int, reserve: int, pixels: int, tile: bool
) -> str:
    """The fastest placement that fits: resident (weights next to the DiT step reserve),
    else offload (weights and one pass's activations once the step released its
    buffers), else ring."""
    act = vae_act_bytes(pixels, tile)
    if weights + reserve + act <= budget:
        return "resident"
    if weights + act <= budget - reserve // 2:
        return "offload"
    return "ring"


def choose_vae_plan(
    plan: VaePlan, weights: int, budget: int, reserve: int, pixels: int
) -> VaePlan:
    """`memory='auto'` resolved: the fastest placement that fits untiled (unless `tile`
    is set), else tiled passes with the fastest placement that fits then."""
    untiled = choose_vae_memory(weights, budget, reserve, pixels, plan.tile)
    if plan.tile or untiled != "ring":
        return replace(plan, memory=untiled)
    tiled = choose_vae_memory(weights, budget, reserve, pixels, True)
    return replace(plan, memory=tiled, tile=True)


class _Resident:
    name = "resident"

    def __init__(self, inner: nn.Module, device: torch.device):
        inner.to(device)
        self.h2d_bytes = 0

    def begin(self) -> None:
        return None

    def end(self) -> None:
        return None


class _Host(_Resident):
    """Everything on the host; encode and decode run there, each pass under
    `host_gemm` of the VAE's dtype."""

    name = "cpu"

    def __init__(self, inner: nn.Module, device: torch.device):
        super().__init__(inner, device)
        self.gemm = host_gemm(next(inner.parameters()).dtype)

    def begin(self) -> None:
        self.gemm.__enter__()

    def end(self) -> None:
        self.gemm.__exit__(None, None, None)


class _Offload:
    """Host copy (pinned on CUDA / XPU) of `inner`'s tensors outside `inside` (ids);
    moved in for a pass, released after it."""

    name = "offload"

    def __init__(
        self, inner: nn.Module, device: torch.device, inside: frozenset = frozenset()
    ):
        self.inner, self.device = inner, device
        tensors = list(inner.parameters()) + list(inner.buffers())
        moved = [t for t in tensors if id(t) not in inside]
        self.copies = HostCopies(moved, device, streamed(device))
        self.bytes = self.copies.nbytes

    @property
    def h2d_bytes(self) -> int:
        return self.copies.h2d_bytes

    def begin(self) -> None:
        self.copies.begin()

    def end(self) -> None:
        self.copies.end()


class _Ring:
    """Encoder / decoder blocks through layer rings; the rest offloaded per pass."""

    name = "ring"

    def __init__(self, inner: nn.Module, device: torch.device):
        parts = [getattr(inner, n) for n in ("encoder", "decoder") if hasattr(inner, n)]
        lists = [find_layer_list(m) for m in (parts or [inner])]
        self.streamers = [LayerStreamer(lst, device, slots=2) for lst in lists]
        self.rest = _Offload(
            inner, device, tensor_ids([b for lst in lists for b in lst])
        )

    @property
    def h2d_bytes(self) -> int:
        return sum(s.h2d_bytes for s in self.streamers) + self.rest.h2d_bytes

    def begin(self) -> None:
        self.rest.begin()

    def end(self) -> None:
        for s in self.streamers:
            s.close()
        self.rest.end()


PLACEMENTS = {
    "resident": _Resident,
    "offload": _Offload,
    "ring": _Ring,
    "cpu": _Host,
}


class PlacedVae:
    """A FluxVae under a VaePlan: `encode_images` / `decode` bracket each pass with the
    placement's begin / end, `encode_batches` covers several batches in one pass.
    `stats` counts passes, images, seconds and host-to-device bytes."""

    def __init__(self, vae, plan: VaePlan, device: torch.device):
        self.vae, self.plan = vae, plan
        self.device = (
            torch.device("cpu") if plan.memory == "cpu" else with_index(device)
        )
        vae.mean.data = vae.mean.data.to(self.device)
        vae.std.data = vae.std.data.to(self.device)
        vae.batch = plan.batch
        if plan.tile:
            for name in ("enable_tiling", "enable_slicing"):
                getattr(vae.vae, name, nothing)()
        self.placement = PLACEMENTS[plan.memory](vae.vae, self.device)
        if compiles(plan, self.device) and hasattr(vae, "accelerate"):
            vae.accelerate(channels_last=True, compile=True)
        self.stats = {"passes": 0, "images": 0, "seconds": 0.0, "h2d_bytes": 0}
        self.before_pass = self.after_pass = nothing

    @property
    def dtype(self) -> torch.dtype:
        return self.vae.dtype

    def to(self, *args, **kwargs) -> "PlacedVae":
        """Placement is the plan's; `.to()` (the pipeline's residency) keeps it."""
        return self

    def set_timeline(self, timeline) -> None:
        self.before_pass = lambda: timeline.mark("step")
        self.after_pass = lambda: timeline.mark("vae")

    def _run(self, fn, count: int):
        start = time.perf_counter()
        self.before_pass()
        self.placement.begin()
        try:
            with on_device(torch.device(self.device)):
                out = fn()
        finally:
            self.placement.end()
        self.after_pass()
        self.stats["passes"] += 1
        self.stats["images"] += count
        self.stats["seconds"] += time.perf_counter() - start
        self.stats["h2d_bytes"] = self.placement.h2d_bytes
        return out

    def encode_images(self, images: Sequence, hw: Tensor, generator=None) -> Tensor:
        """EncodeImages under the placement (one pass)."""
        return self._run(
            lambda: self.vae.encode_images(images, hw, generator), len(images)
        )

    def encode_batches(self, batches: Sequence[tuple]) -> list[Tensor]:
        """[(images, hw), ...] -> one packed latent tensor per batch, one pass."""
        n = sum(len(images) for images, _ in batches)
        return self._run(
            lambda: [self.vae.encode_images(i, hw) for i, hw in batches], n
        )

    def decode(self, latents: Tensor) -> Tensor:
        """Decode under the placement (one pass)."""
        return self._run(lambda: self.vae.decode(latents.to(self.device)), 1)


def place_vae(
    vae,
    spec: dict | VaePlan | None,
    device: torch.device | str,
    reserve: int = 0,
    pixels: int = 1024 * 1024,
) -> PlacedVae:
    """`vae` (a FluxVae, loaded on the host) placed by its plan; `memory='auto'` picks
    the fastest placement that fits the device's free memory next to `reserve` bytes
    (`choose_vae_plan`: tiled passes when nothing fits untiled), 'cpu' on the host."""
    plan = vae_plan(spec)
    device = torch.device(device)
    if plan.memory == "auto":
        if streamed(device):
            free = free_bytes(with_index(device))
            plan = choose_vae_plan(plan, vae_bytes(vae.vae), free, reserve, pixels)
        else:
            on_host = device.type == "cpu"
            plan = replace(plan, memory="cpu" if on_host else "resident")
    return PlacedVae(vae, plan, device)
