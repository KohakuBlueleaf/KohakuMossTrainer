"""Weight EMA as explicit fp32 buffers, updated after each successful optimiser step.

`ConstantEMA` keeps one average per decay; `PowerEMA` keeps EDM2 power-function
averages (beta_t = (1 - 1/t)^(gamma + 1)) per sigma_rel, whose snapshots allow post-hoc
reconstruction (`posthoc.py`). Each buffer lives on its parameter's device unless the
spec names one (host masters of streamed blocks: host buffers).
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.offload.accel import module, streamed
from kohakumoss_common.calls import nothing
from kohakumoss_common.runs import runs_within


def gamma_from_sigma_rel(sigma_rel: float, iters: int = 60) -> float:
    """Solve sigma_rel^2 = (g+1) / ((g+2)^2 (g+3)) for gamma (EDM2, eq. 3): the largest
    root of g^3 + 7 g^2 + (16 - t) g + 12 - t, t = sigma_rel^-2, by float32 Newton
    from g = sqrt(t), right of the root where the cubic is convex and increasing."""
    t = torch.tensor(sigma_rel, dtype=torch.float32).pow(-2)
    g = t.sqrt()
    for _ in range(iters):
        f = ((g + 7.0) * g + (16.0 - t)) * g + (12.0 - t)
        df = (3.0 * g + 14.0) * g + (16.0 - t)
        g = g - f / df
    return float(g)


class NoEMA:
    """EMA disabled."""

    tags: tuple[str, ...] = ()

    def update(self, step: int) -> None:
        return None

    def state_dict(self) -> dict[str, Tensor]:
        return {}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        return None

    def averaged(self, tag: str) -> dict[str, Tensor]:
        raise KeyError(tag)


CHUNK_NUMEL = 64 * 1024 * 1024  # fp32 elements of converted copies alive at once


class _HostStage:
    """Two pinned fp32 host buffers of `numel` elements: an accelerator chunk is copied
    into one (non-blocking, one event) while the other chunk is averaged."""

    def __init__(self, numel: int, device: torch.device):
        self.dm = module(device)
        self.device = device
        self.flats = [
            torch.empty(numel, dtype=torch.float32, pin_memory=True) for _ in range(2)
        ]
        self.events = [None, None]

    def start(self, which: int, params: list[Tensor]) -> list[Tensor]:
        """Queue the D2H of `params` into buffer `which`; its fp32 views."""
        views, offset = [], 0
        for p in params:
            n = p.numel()
            view = self.flats[which][offset : offset + n].view(p.shape)
            view.copy_(p.detach(), non_blocking=True)
            views.append(view)
            offset += n
        ev = self.dm.Event()
        ev.record(self.dm.current_stream(self.device))
        self.events[which] = ev
        return views

    def wait(self, which: int) -> None:
        self.events[which].synchronize()


class _BufferEMA:
    """fp32 average buffers. Parameters already fp32 on the buffer's device are lerped
    in place in one foreach call; the others are converted in chunks of at most
    `chunk_numel` elements, so one chunk of fp32 copies is alive at a time; each chunk
    is converted once per update and averaged into every tag. Accelerator parameters
    averaged in host buffers go through two pinned staging chunks (`_HostStage`)."""

    def __init__(
        self,
        named: list[tuple[str, nn.Parameter]],
        tags,
        device,
        chunk_numel: int = CHUNK_NUMEL,
    ):
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        self.tags = tuple(tags)
        self.before_read: Callable[[], None] = nothing
        self.buffers = {
            tag: [
                p.detach().to(device=device or p.device, dtype=torch.float32, copy=True)
                for p in self.params
            ]
            for tag in self.tags
        }
        ref = next(iter(self.buffers.values()), [])
        same = [
            i
            for i, (p, b) in enumerate(zip(self.params, ref))
            if p.dtype == torch.float32 and p.device == b.device
        ]
        rest = [i for i in range(len(self.params)) if i not in set(same)]
        self.direct = same
        sizes = [p.numel() for p in self.params]
        self.converted = runs_within(rest, sizes.__getitem__, max(int(chunk_numel), 1))
        staged = [
            i
            for i in rest
            if streamed(self.params[i].device) and ref[i].device.type == "cpu"
        ]
        self.staged = set(staged)
        self.stage = None
        if staged:
            largest = max(
                sum(self.params[i].numel() for i in c) for c in self.converted
            )
            self.stage = _HostStage(largest, self.params[staged[0]].device)

    def _convert(self, chunk: list[int]) -> list[Tensor]:
        """fp32 copies on the buffers' devices (asynchronous only onto an accelerator,
        where the averaging that reads them follows on the same stream)."""
        ref = self.buffers[self.tags[0]]
        return [
            self.params[i]
            .detach()
            .to(ref[i].device, torch.float32, non_blocking=streamed(ref[i].device))
            for i in chunk
        ]

    def _lerp_all(self, weights: list[float]) -> None:
        """Average every parameter into every tag's buffers (`weights` per tag), once
        `before_read()` returned (the masters' write-back has landed)."""
        self.before_read()
        if self.direct:
            src = [self.params[i].detach() for i in self.direct]
            for tag, w in zip(self.tags, weights):
                bufs = self.buffers[tag]
                torch._foreach_lerp_([bufs[i] for i in self.direct], src, w)
        chunks = self.converted
        staged = [all(i in self.staged for i in c) for c in chunks]
        ahead = None
        for n, chunk in enumerate(chunks):
            if staged[n]:
                which = n % 2
                src = ahead if ahead is not None else self._start(chunk, which)
                nxt = n + 1
                ahead = (
                    self._start(chunks[nxt], nxt % 2)
                    if nxt < len(chunks) and staged[nxt]
                    else None
                )
                self.stage.wait(which)
            else:
                src = self._convert(chunk)
            for tag, w in zip(self.tags, weights):
                bufs = self.buffers[tag]
                torch._foreach_lerp_([bufs[i] for i in chunk], src, w)

    def _start(self, chunk: list[int], which: int) -> list[Tensor]:
        return self.stage.start(which, [self.params[i] for i in chunk])

    def state_dict(self) -> dict[str, Tensor]:
        return {
            f"{tag}/{name}": buf
            for tag in self.tags
            for name, buf in zip(self.names, self.buffers[tag])
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for tag in self.tags:
            for name, buf in zip(self.names, self.buffers[tag]):
                buf.copy_(state[f"{tag}/{name}"])

    def averaged(self, tag: str) -> dict[str, Tensor]:
        return dict(zip(self.names, self.buffers[tag]))


class BackgroundEMA:
    """`inner`'s updates on one worker thread, each overlapping the next step's
    forward: `join()` waits for the running update (the driver calls it before each
    backward); the next update, a state read and a load each join first."""

    def __init__(self, inner):
        self.inner = inner
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="ema")
        self.pending = None
        accel = [p.device for p in getattr(inner, "params", []) if streamed(p.device)]
        self.device = accel[0] if accel else None

    def _update(self, step: int) -> None:
        if self.device is None:
            self.inner.update(step)
            return
        with module(self.device).device(self.device):
            self.inner.update(step)

    @property
    def tags(self) -> tuple[str, ...]:
        return self.inner.tags

    @property
    def gammas(self) -> tuple[float, ...]:
        return getattr(self.inner, "gammas", ())

    @property
    def params(self) -> list[Tensor]:
        return self.inner.params

    @property
    def buffers(self) -> dict[str, list[Tensor]]:
        """The average buffers, once the running update has landed."""
        self.join()
        return self.inner.buffers

    @property
    def before_read(self) -> Callable[[], None]:
        return self.inner.before_read

    @before_read.setter
    def before_read(self, fn: Callable[[], None]) -> None:
        self.inner.before_read = fn

    def join(self) -> None:
        if self.pending is not None:
            pending, self.pending = self.pending, None
            pending.result()

    def update(self, step: int) -> None:
        self.join()
        self.pending = self.pool.submit(self._update, step)

    def state_dict(self) -> dict[str, Tensor]:
        self.join()
        return self.inner.state_dict()

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.join()
        self.inner.load_state_dict(state)

    def averaged(self, tag: str) -> dict[str, Tensor]:
        self.join()
        return self.inner.averaged(tag)


def host_averaged(ema) -> bool:
    """Whether any of `ema`'s buffers is a host buffer (a CPU pass every update)."""
    buffers = getattr(ema, "buffers", {})
    return any(b.device.type == "cpu" for bufs in buffers.values() for b in bufs)


def _warm_decay(decay: float, t: int) -> float:
    return min(decay, (1 + t) / (10 + t))


def _fixed_decay(decay: float, t: int) -> float:
    return decay


class ConstantEMA(_BufferEMA):
    """ema <- decay * ema + (1 - decay) * p, with optional (1+t)/(10+t) warmup."""

    def __init__(
        self,
        named,
        decays: tuple[float, ...],
        warmup: bool = True,
        device=None,
        chunk_numel: int = CHUNK_NUMEL,
    ):
        super().__init__(named, [f"ema{d:g}" for d in decays], device, chunk_numel)
        self.decays = tuple(decays)
        rule = _warm_decay if warmup else _fixed_decay
        self.rules = tuple(partial(rule, d) for d in self.decays)

    @torch.no_grad()
    def update(self, step: int) -> None:
        t = step + 1
        self._lerp_all([1.0 - rule(t) for rule in self.rules])


class PowerEMA(_BufferEMA):
    """EDM2 power-function EMA per sigma_rel."""

    def __init__(
        self,
        named,
        sigma_rels: tuple[float, ...],
        device=None,
        chunk_numel: int = CHUNK_NUMEL,
    ):
        super().__init__(
            named, [f"power{s:g}" for s in sigma_rels], device, chunk_numel
        )
        self.sigma_rels = tuple(sigma_rels)
        self.gammas = tuple(gamma_from_sigma_rel(s) for s in sigma_rels)

    @torch.no_grad()
    def update(self, step: int) -> None:
        t = step + 1
        self._lerp_all([1.0 - (1.0 - 1.0 / t) ** (g + 1.0) for g in self.gammas])


def build_ema(named, spec, device=None):
    """EMA knob: 0 / None -> off; float or list -> ConstantEMA; dict with kind."""
    match spec:
        case None | 0 | 0.0:
            return NoEMA()
        case float() | int():
            return ConstantEMA(named, (float(spec),), device=device)
        case list() | tuple():
            return ConstantEMA(named, tuple(float(d) for d in spec), device=device)
        case {"kind": "power", **rest}:
            return PowerEMA(
                named, tuple(rest["sigma_rel"]), device=rest.get("device", device)
            )
        case {"kind": "constant", **rest}:
            return ConstantEMA(
                named,
                (
                    tuple(rest["decay"])
                    if isinstance(rest["decay"], (list, tuple))
                    else (rest["decay"],)
                ),
                warmup=rest.get("warmup", True),
                device=rest.get("device", device),
            )
        case _:
            raise ValueError(f"invalid EMA spec {spec!r}")


def ema_is_power(ema) -> bool:
    return isinstance(getattr(ema, "inner", ema), PowerEMA)


def decay_count(spec) -> int:
    """How many averages the EMA knob keeps (0: off)."""
    match spec:
        case None | 0 | 0.0:
            return 0
        case float() | int():
            return 1
        case list() | tuple():
            return len(spec)
        case {"kind": "power", **rest}:
            return len(tuple(rest["sigma_rel"]))
        case {"kind": "constant", **rest}:
            decay = rest["decay"]
            return len(decay) if isinstance(decay, (list, tuple)) else 1
    raise ValueError(f"invalid EMA spec {spec!r}")
