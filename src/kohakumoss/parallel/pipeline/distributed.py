"""Process-group pipeline parallelism over torch.distributed.pipelining: contiguous
layer-range stages with static micro-batch shapes. Every rank holds the step's
micro-batch layouts; only (x, modulation, head modulation) cross a stage boundary. See
docs/guides/multi-gpu.md."""

import math
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import pipelining

from kohakumoss.memory.runners import PlainRunner
from kohakumoss.parallel.pipeline.parts import PartsCtx
from kohakumoss.parallel.telemetry.timing import build_clock
from kohakumoss.types.seqinfo import SeqInfo, build_seqinfo
from kohakumoss_common.calls import nothing
from kohakumoss_common.weak import weak_method
from kohakumoss_data.config import DataConfig, pipeline_problems

SCHEDULES = {
    "gpipe": "ScheduleGPipe",
    "1f1b": "Schedule1F1B",
    "interleaved-1f1b": "ScheduleInterleaved1F1B",
    "looped-bfs": "ScheduleLoopedBFS",
    "interleaved-zb": "ScheduleInterleavedZeroBubble",
    "zbv": "ScheduleZBVZeroBubble",
    "dualpipe-v": "ScheduleDualPipeV",
}
SINGLE = ("gpipe", "1f1b")
V_SHAPED = ("zbv", "dualpipe-v")
LOOPED = ("interleaved-1f1b", "interleaved-zb")
SPLIT_BACKWARD = ("interleaved-zb", "zbv", "dualpipe-v")
PROBE_TEXT = 300


def retained_backward(schedule: str) -> None:
    """Compile backwards without donated buffers when `schedule` runs a stage's
    backward as separate input and weight passes over one retained graph; set as the
    config default, so the autograd threads see it too (overrides are per thread)."""
    if schedule in SPLIT_BACKWARD:
        torch._functorch.config._config["donated_buffer"].default = False


def schedule_class(name: str):
    cls = getattr(pipelining, SCHEDULES.get(name, ""), None)
    if cls is None:
        raise ValueError(f"pipeline schedule {name!r} not available in this torch")
    return cls


def stages_per_rank(schedule: str, virtual: int = 2) -> int:
    """Stages each rank holds: 1 for gpipe / 1f1b, 2 for the V schedules, `virtual`
    for the looped ones."""
    if schedule in SINGLE:
        return 1
    if schedule in V_SHAPED:
        return 2
    return virtual


def stage_ids(rank: int, world: int, schedule: str, per_rank: int) -> list[int]:
    """Virtual stage indices held by `rank` (V placement for zbv / dualpipe-v)."""
    if schedule in SINGLE:
        return [rank]
    if schedule in V_SHAPED:
        return [rank, 2 * world - 1 - rank]
    return [rank + k * world for k in range(per_rank)]


def stage_owner(stage: int, world: int, schedule: str) -> int:
    """The rank holding virtual stage `stage`."""
    if schedule in V_SHAPED:
        return stage if stage < world else 2 * world - 1 - stage
    return stage % world


def schedule_problems(schedule: str, world: int, microbatches: int) -> list[str]:
    """What keeps `schedule` from running `microbatches` over `world` ranks."""
    p = []
    if schedule not in SCHEDULES:
        p.append(f"PIPELINE_SCHEDULE={schedule!r} not in {tuple(SCHEDULES)}")
    if microbatches < world:
        p.append(f"MICROBATCHES={microbatches} < {world} pipeline ranks")
    rounds = max(1, microbatches // world)
    if schedule in LOOPED and microbatches % rounds:
        p.append(
            f"PIPELINE_SCHEDULE={schedule!r} runs MICROBATCHES in {rounds} rounds: "
            f"{microbatches} is not a multiple of {rounds}"
        )
    if schedule == "dualpipe-v" and microbatches < 2 * world:
        p.append(f"dualpipe-v needs MICROBATCHES >= {2 * world} (its stage count)")
    return p


@dataclass
class MicroBatch:
    """One micro-batch: packed inputs, layout sizes (host), target, weights.

    `img_keep[i]` (host int64 indices into sample i's row-major grid, or None for all)
    is the token-drop subset; `img` / `target` then hold only the kept tokens."""

    img: Tensor
    text: Tensor
    sigma: Tensor
    hw: Tensor
    text_len: Tensor
    target: Tensor
    weight: Tensor
    img_keep: list[Tensor | None] | None = None

    def img_counts(self) -> Tensor:
        """Image tokens of each sample after token drop (int64, host)."""
        full = (self.hw[:, 0] * self.hw[:, 1]).to(torch.int64).cpu()
        if self.img_keep is None:
            return full
        return torch.tensor(
            [int(n) if k is None else k.numel() for n, k in zip(full, self.img_keep)],
            dtype=torch.int64,
        )


HOST_FIELDS = ("hw", "text_len", "img_keep")


def microbatch_to(mb: MicroBatch, device) -> MicroBatch:
    """Data tensors to `device`; the layout fields (`hw`, `text_len`, `img_keep`) stay
    on the host."""
    moved = {
        f.name: (
            getattr(mb, f.name)
            if f.name in HOST_FIELDS
            else getattr(mb, f.name).to(device, non_blocking=True)
        )
        for f in fields(MicroBatch)
    }
    return MicroBatch(**moved)


@dataclass(frozen=True)
class PipeShape:
    """Static micro-batch shape every stage is built for: `tokens` joint tokens and
    `max_samples` samples per micro-batch, `microbatches` per step. Padding tokens are
    split into zero-weight samples of at most `pad_segment` tokens; `sample_tokens` is
    a typical sample's joint length (the stage-cost probe)."""

    tokens: int
    max_samples: int
    dim: int
    microbatches: int
    rope_axes: int = 4
    pad_segment: int = 512
    sample_tokens: int = 1024

    @property
    def slots(self) -> int:
        """Sample rows of every micro-batch (real samples plus padding segments)."""
        return self.max_samples + math.ceil(self.tokens / self.pad_segment)

    @classmethod
    def from_knobs(cls, knobs: dict, dim: int, rope_axes: int = 4) -> "PipeShape":
        """The shape the data planner guarantees: MAX_TOKENS joint tokens and
        MAX_SAMPLES samples per micro-batch, MICROBATCHES per step. Raises when the
        data knobs do not make those hard bounds (`pipeline_problems`)."""
        cfg = DataConfig.from_knobs(knobs)
        problems = pipeline_problems(cfg)
        if problems:
            raise ValueError("pipeline micro-batch shape: " + "; ".join(problems))
        image = max(1, cfg.target_area // cfg.latent_factor**2)
        return cls(
            tokens=cfg.max_tokens,
            max_samples=cfg.max_samples,
            dim=dim,
            microbatches=knobs["MICROBATCHES"],
            rope_axes=rope_axes,
            sample_tokens=min(cfg.max_tokens, image + PROBE_TEXT),
        )

    def probe(self) -> tuple[Tensor, Tensor]:
        """(hw, text_len) of a micro-batch of typical samples filling the shape."""
        n = max(1, min(self.max_samples, self.tokens // self.sample_tokens))
        text = min(PROBE_TEXT, self.sample_tokens - 1)
        hw = torch.tensor([[1, self.sample_tokens - text]] * n)
        return hw, torch.tensor([text] * n)


def padding(used: int, n: int, shape: PipeShape) -> list[int]:
    """Token counts of the padding samples filling `used` tokens of `n` samples up to
    the shape: segments of at most `pad_segment`, then empty slots."""
    pad = shape.tokens - used
    if pad < 0 or n > shape.max_samples:
        raise ValueError(f"micro-batch exceeds pipe shape ({used} tokens, {n} samples)")
    k = math.ceil(pad / shape.pad_segment)
    sizes = [pad // k + (1 if i < pad % k else 0) for i in range(k)] if k else []
    return sizes + [0] * (shape.slots - n - len(sizes))


def padded_layout(hw, text_len, img_keep, used: int, shape: PipeShape) -> SeqInfo:
    """Layout of a micro-batch plus its padding samples; kept image tokens keep their
    original grid positions."""
    n = hw.shape[0]
    pads = padding(used, n, shape)
    hw = torch.cat([hw.to(torch.int64), torch.tensor([[1, s] for s in pads])])
    empty = torch.zeros(len(pads), dtype=torch.int64)
    text = torch.cat([text_len.to(torch.int64), empty])
    keep = list(img_keep) + [None] * len(pads) if img_keep is not None else None
    return build_seqinfo(hw, text, shape.rope_axes, img_keep=keep)


def padded_seqinfo(mb: MicroBatch, shape: PipeShape) -> SeqInfo:
    """Layout of `mb` plus its padding samples (`padded_layout`)."""
    used = int(mb.img_counts().sum() + mb.text_len.sum())
    return padded_layout(mb.hw, mb.text_len, mb.img_keep, used, shape)


def _owned(t: Tensor) -> Tensor:
    """`t` itself when it owns its storage, else a copy (a stage output the schedule
    detaches in place must not be a view)."""
    return t if t._base is None else t.clone()


def _same(t: Tensor) -> Tensor:
    return t


def _pad_rows(t: Tensor, rows: int) -> Tensor:
    if t.shape[0] == rows:
        return t
    return torch.cat([t, t.new_zeros(rows - t.shape[0], *t.shape[1:])])


class StageModule(nn.Module):
    """Layers [lo, hi) of `parts`, plus its prologue (first) / epilogue modules (last).
    `forward(..., mb=k)` runs micro-batch k of the step set by `set_step` and returns
    (x in the boundary dtype, modulation, head modulation); `own_outputs`: none of
    them is a view (the split-backward schedules detach stage outputs in place)."""

    def __init__(
        self,
        parts,
        lo: int,
        hi: int,
        first: bool,
        last: bool,
        runner=None,
        boundary: torch.dtype = torch.bfloat16,
        own_outputs: bool = False,
    ):
        super().__init__()
        self.lo, self.hi, self.first, self.last = lo, hi, first, last
        self._own = _owned if own_outputs else _same
        self.layers = nn.ModuleDict({str(i): parts.layers[i] for i in range(lo, hi)})
        self.parts = [parts]
        if first:
            self.head_parts = nn.ModuleList(parts.prologue_modules())
        if last:
            self.tail = nn.ModuleList(parts.epilogue_modules())
        self.runner = (runner or PlainRunner()).bind(parts.layers)
        self.seqs: list[SeqInfo] = []
        self.inputs: list[MicroBatch] = []
        self.register_buffer("bands", parts.bands.detach().clone(), persistent=False)
        self.residual = parts.residual_scale
        self.boundary = boundary
        self._blocks = tuple((i, self.layers[str(i)]) for i in range(lo, hi))
        self._forward = (
            StageModule._forward_first if first else StageModule._forward_mid
        )

    def set_step(self, seqs: list[SeqInfo], inputs: list[MicroBatch]) -> None:
        self.seqs, self.inputs = seqs, inputs

    def _run(self, x: Tensor, ctx: PartsCtx) -> Tensor:
        run = self.runner.run
        for i, block in self._blocks:
            x = run(i, block, x, ctx)
        return x.to(self.boundary)

    def forward(self, *args, mb: int):
        return self._forward(self, *args, mb=mb)

    def _forward_first(self, token: Tensor, mb: int):
        x, ctx = self._prologue(self.inputs[mb], self.seqs[mb])
        own = self._own
        return own(self._run(x, ctx)), own(ctx.mod), own(ctx.head)

    def _forward_mid(self, x: Tensor, mod: Tensor, head: Tensor, mb: int):
        ctx = PartsCtx(self.seqs[mb], mod, self.bands, self.residual, head)
        own = self._own
        return own(self._run(x, ctx)), own(mod), own(head)

    def ctx(self, seq: SeqInfo, mod: Tensor, head: Tensor) -> PartsCtx:
        return PartsCtx(seq, mod, self.bands, self.residual, head)

    def _prologue(self, data: MicroBatch, seq: SeqInfo):
        img = _pad_rows(data.img, seq.n_img)
        text = _pad_rows(data.text, seq.n_txt)
        sigma = _pad_rows(data.sigma, seq.batch)
        return self.parts[0].prologue(img, text, sigma, seq)


class TimedStage(pipelining.PipelineStage):
    """PipelineStage whose chunk forwards / backwards are bracketed by device events
    of `clock` (`fwd`, `bwd`, `bwd_w`); `on_chunk(k)` runs before each of them for
    micro-batch k."""

    def __init__(self, *args, clock, on_chunk: Callable[[int], None], **kwargs):
        super().__init__(*args, **kwargs)
        self.clock = clock
        self.on_chunk = on_chunk

    def forward_one_chunk(self, fwd_chunk_id: int, *args, **kwargs):
        self.on_chunk(fwd_chunk_id)
        mark = self.clock.start()
        out = super().forward_one_chunk(fwd_chunk_id, *args, **kwargs)
        self.clock.stop("fwd", mark)
        return out

    def backward_one_chunk(self, bwd_chunk_id: int, *args, **kwargs):
        self.on_chunk(bwd_chunk_id)
        mark = self.clock.start()
        out = super().backward_one_chunk(bwd_chunk_id, *args, **kwargs)
        self.clock.stop("bwd", mark)
        return out

    def backward_weight_one_chunk(self, bwd_chunk_id: int, *args, **kwargs):
        self.on_chunk(bwd_chunk_id)
        mark = self.clock.start()
        out = super().backward_weight_one_chunk(bwd_chunk_id, *args, **kwargs)
        self.clock.stop("bwd_w", mark)
        return out


class PipelineEngine:
    """This rank's stages and schedule; `step(microbatches, denom)` runs forward +
    backward of the step and returns its loss on every rank.

    `objective(pred, data, seg)` is the loss of one micro-batch (`seg`: int32 sample
    index of each real image token); `bounds` holds one [lo, hi) layer range per
    virtual stage. `capture(sigma, seq)` draws a micro-batch's adapter step state from
    its padded layout (in micro-batch order, once per step) and `install(state)` puts
    it back before each of that micro-batch's chunk forwards / backwards."""

    def __init__(
        self,
        parts,
        rank: int,
        world: int,
        device: torch.device,
        shape: PipeShape,
        bounds: list[tuple[int, int]],
        schedule: str,
        objective: Callable,
        compute_dtype: torch.dtype,
        runner=None,
        per_rank: int = 1,
        group=None,
    ):
        problems = schedule_problems(schedule, world, shape.microbatches)
        if problems:
            raise ValueError("; ".join(problems))
        self.rank, self.world, self.device, self.shape = rank, world, device, shape
        self.dtype = compute_dtype
        self.boundary = getattr(parts, "residual_dtype", compute_dtype)
        self.cond_grad = parts.conditioning_trainable()
        self.group = group
        self.capture: Callable = nothing
        self.install: Callable = nothing
        self.states: list = [None] * shape.microbatches
        ids = stage_ids(rank, world, schedule, per_rank)
        n_stages = world * len(ids)
        if len(bounds) != n_stages:
            raise ValueError(f"need {n_stages} layer ranges, got {len(bounds)}")
        self.clock = build_clock(device)
        self.modules_: list[StageModule] = []
        stages = []
        for sid in ids:
            lo, hi = bounds[sid]
            mod = StageModule(
                parts,
                lo,
                hi,
                sid == 0,
                sid == n_stages - 1,
                runner,
                self.boundary,
                own_outputs=schedule in SPLIT_BACKWARD,
            )
            mod.to(device)
            self.modules_.append(mod)
            stages.append(self._stage(mod, sid, n_stages, group))
        self.bounds = list(bounds)
        self.first = any(m.first for m in self.modules_)
        self.last = any(m.last for m in self.modules_)
        self.tail = next((m for m in self.modules_ if m.last), None)
        self.loss_rank = stage_owner(n_stages - 1, world, schedule)
        if group is not None:
            self.loss_rank = dist.get_global_rank(group, self.loss_rank)
        self.objective = objective
        self.denom = 1.0
        self.data: list[MicroBatch] = []
        m = shape.microbatches
        token = torch.zeros(1, dtype=torch.int64, device=device)
        self.call = {"kwarg_mbs": [{"mb": i} for i in range(m)]}
        if self.first:
            self.call["arg_mbs"] = [(token,)] * m
        if self.last:
            self.call["target_mbs"] = list(range(m))
        self._check = PipelineEngine._check_layouts
        self.pad_tokens = 0
        self.real_tokens = 0
        cls = schedule_class(schedule)
        target = stages[0] if schedule in SINGLE else stages
        loss = weak_method(self._loss)
        self.schedule = cls(target, m, loss_fn=loss, scale_grads=False)

    def _example(self, first: bool):
        """Static stage metadata: `(token,)` into the first stage, else `(x, mod,
        head)`; x always carries a gradient, the modulations when a conditioning
        parameter trains."""
        s, d = self.shape, self.device
        if first:
            return (torch.zeros(1, dtype=torch.int64, device=d),)
        x = torch.empty(s.tokens, s.dim, dtype=self.boundary, device=d)
        mod = torch.empty(s.slots, 6, s.dim, dtype=torch.float32, device=d)
        head = torch.empty(s.slots, 2, s.dim, dtype=torch.float32, device=d)
        g = self.cond_grad
        return (x.requires_grad_(), mod.requires_grad_(g), head.requires_grad_(g))

    def _stage(self, mod: StageModule, sid: int, n: int, group):
        return TimedStage(
            mod,
            sid,
            n,
            self.device,
            input_args=self._example(mod.first),
            output_args=self._example(False),
            group=group,
            clock=self.clock,
            on_chunk=weak_method(self._install_chunk),
        )

    def _install_chunk(self, k: int) -> None:
        self.install(self.states[k])

    def _loss(self, output, k: int) -> Tensor:
        mark = self.clock.start()
        x, mod, head = output
        data, tail = self.data[k], self.tail
        seq = tail.seqs[k]
        n = data.target.shape[0]
        pred = tail.parts[0].epilogue(x, tail.ctx(seq, mod, head))[:n]
        loss = self.objective(pred, data, seq.seg_img[:n]) / self.denom
        self.clock.stop("loss", mark)
        return loss

    def _check_layouts(self, microbatches: list[MicroBatch]) -> None:
        """Every rank drew the same micro-batches (sizes and sigma bits); raises on
        every rank otherwise. Runs on the first step only."""
        rows = []
        for mb in microbatches:
            bits = mb.sigma.float().contiguous().view(torch.int32).to(torch.int64)
            sizes = torch.stack(
                [mb.hw.to(torch.int64).sum(), mb.text_len.to(torch.int64).sum()]
            )
            rows.append(torch.cat([sizes.to(self.device), bits.sum().reshape(1)]))
        mine = torch.stack(rows)
        world = dist.get_world_size(self.group)
        every = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(every, mine, group=self.group)
        if any(not torch.equal(mine, other) for other in every):
            raise RuntimeError(
                "pipeline ranks drew different micro-batches (layout / sigma digests "
                "differ): every stage must see the same step"
            )
        self._check = nothing

    def step(self, microbatches: list[MicroBatch], denom: float) -> Tensor:
        """One forward + backward over the step; the step's loss (summed over its
        micro-batches, still times the loss scale) on every rank."""
        self._check(self, microbatches)
        seqs = [padded_seqinfo(mb, self.shape).to(self.device) for mb in microbatches]
        moved = [microbatch_to(mb, self.device) for mb in microbatches]
        for m in self.modules_:
            m.set_step(seqs, moved)
        self.states = [
            self.capture(_pad_rows(mb.sigma, seq.batch), seq)
            for mb, seq in zip(moved, seqs)
        ]
        self.data, self.denom = moved, denom
        real = sum(int(mb.img_counts().sum() + mb.text_len.sum()) for mb in moved)
        self.real_tokens += real
        self.pad_tokens += self.shape.tokens * len(moved) - real
        losses: list[Tensor] | None = [] if self.last else None
        self.clock.begin()
        self.schedule.step(**self.call, losses=losses, return_outputs=False)
        self.clock.end()
        buf = torch.zeros(1, device=self.device)
        if losses:
            buf[0] = torch.stack([v.detach().float() for v in losses]).sum()
        dist.broadcast(buf, src=self.loss_rank, group=self.group)
        return buf[0]

    def metrics(self) -> dict[str, float]:
        """The last step's device time of this rank's stages: forward, backward and
        loss compute, the step span, the rest of the span (`pipe_idle_ms`: bubble
        plus exposed transfers) and its fraction; the padding share of the tokens
        since the last call."""
        t = self.clock.read()
        out = {}
        if t:
            compute = sum(v for k, v in t.items() if k != "wall_ms")
            out = {f"pipe_{k}": v for k, v in t.items()}
            out["pipe_compute_ms"] = compute
            out["pipe_idle_ms"] = t["wall_ms"] - compute
            out["pipe_bubble_frac"] = out["pipe_idle_ms"] / max(t["wall_ms"], 1e-9)
        total = self.real_tokens + self.pad_tokens
        if total:
            out["pipe_pad_frac"] = self.pad_tokens / total
        self.real_tokens = self.pad_tokens = 0
        return out

    def parameters(self):
        for m in self.modules_:
            yield from m.parameters()


class TokenLoss:
    """Loss of one micro-batch on the last stage: the configured token loss
    (`loss_fn(pred, target, seg, weight, mask)`, as TrainStep calls it) over its real
    image tokens, times the loss scale (`scaler`, set once the optimiser bundle
    exists)."""

    def __init__(self, loss_fn: Callable, scaler=None):
        self.token_loss = loss_fn
        self.scaler = scaler

    def __call__(self, pred: Tensor, data: MicroBatch, seg: Tensor) -> Tensor:
        loss = self.token_loss(pred, data.target, seg, data.weight, None)
        return loss * self.scaler.scale
