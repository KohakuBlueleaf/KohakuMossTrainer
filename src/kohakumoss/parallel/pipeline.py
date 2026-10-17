"""Process-group pipeline parallelism over torch.distributed.pipelining: contiguous
layer-range stages with static micro-batch shapes. See docs/guides/multi-gpu.md."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import pipelining

from kohakumoss.parallel.parts import PartsCtx
from kohakumoss.training.memory.runners import PlainRunner
from kohakumoss.types.seqinfo import SeqInfo, build_seqinfo
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
_STAGE_ARGS = set(inspect.signature(pipelining.PipelineStage.__init__).parameters)


def schedule_class(name: str):
    cls = getattr(pipelining, SCHEDULES.get(name, ""), None)
    if cls is None:
        raise ValueError(f"pipeline schedule {name!r} not available in this torch")
    return cls


def stage_ids(rank: int, world: int, schedule: str, per_rank: int) -> list[int]:
    """Virtual stage indices held by `rank` (V placement for zbv / dualpipe-v)."""
    if schedule in SINGLE:
        return [rank]
    if schedule in V_SHAPED:
        return [rank, 2 * world - 1 - rank]
    return [rank + k * world for k in range(per_rank)]


@dataclass
class MicroBatch:
    """One micro-batch on the host: packed inputs, layout sizes, target, weights.

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


@dataclass(frozen=True)
class PipeShape:
    """Static micro-batch shape every stage is built for."""

    tokens: int
    max_samples: int
    dim: int
    microbatches: int
    rope_axes: int = 4

    @classmethod
    def from_knobs(cls, knobs: dict, dim: int, rope_axes: int = 4) -> "PipeShape":
        """The shape the data planner guarantees: MAX_TOKENS joint tokens and
        MAX_SAMPLES samples per micro-batch, MICROBATCHES per step. Raises when the
        data knobs do not make those hard bounds (`pipeline_problems`)."""
        cfg = DataConfig.from_knobs(knobs)
        problems = pipeline_problems(cfg)
        if problems:
            raise ValueError("pipeline micro-batch shape: " + "; ".join(problems))
        return cls(
            tokens=cfg.max_tokens,
            max_samples=cfg.max_samples,
            dim=dim,
            microbatches=knobs["MICROBATCHES"],
            rope_axes=rope_axes,
        )


def padded_seqinfo(mb: MicroBatch, shape: PipeShape) -> SeqInfo:
    """Layout of `mb` plus one padding sample filling up to `shape.tokens` tokens.
    Kept image tokens (`mb.img_keep`) keep their original grid positions."""
    n = mb.hw.shape[0]
    used = int(mb.img_counts().sum() + mb.text_len.sum())
    pad = shape.tokens - used
    if pad < 0 or n > shape.max_samples:
        raise ValueError(f"micro-batch exceeds pipe shape ({used} tokens)")
    n_pad = shape.max_samples + 1 - n
    hw = torch.cat([mb.hw, torch.tensor([[1, pad]] + [[1, 0]] * (n_pad - 1))])
    text = torch.cat([mb.text_len, torch.zeros(n_pad, dtype=mb.text_len.dtype)])
    keep = list(mb.img_keep) + [None] * n_pad if mb.img_keep is not None else None
    return build_seqinfo(hw, text, shape.rope_axes, img_keep=keep)


class StageModule(nn.Module):
    """Layers [lo, hi) of `parts`, plus its prologue (first) / epilogue (last)."""

    def __init__(self, parts, lo: int, hi: int, first: bool, last: bool, runner=None):
        super().__init__()
        self.lo, self.hi, self.first, self.last = lo, hi, first, last
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
        self._blocks = tuple((i, self.layers[str(i)]) for i in range(lo, hi))
        self._forward = self._forward_first if first else self._forward_mid
        self.calls = 0

    def set_step(self, seqs: list[SeqInfo], inputs: list[MicroBatch]) -> None:
        self.seqs, self.inputs = seqs, inputs
        self.calls = 0

    def _run(self, x: Tensor, ctx: PartsCtx) -> Tensor:
        run = self.runner.run
        for i, block in self._blocks:
            x = run(i, block, x, ctx)
        return x

    def _index(self, mb: Tensor) -> int:
        """This forward's micro-batch index (a stage runs its forwards in micro-batch
        order); host-side, no device read."""
        k = self.calls
        if k >= len(self.seqs):
            raise RuntimeError(
                f"stage forward {k} beyond {len(self.seqs)} micro-batches"
            )
        self.calls = k + 1
        return k

    def forward(self, *args):
        return self._forward(*args)

    def _forward_first(self, mb: Tensor):
        k = self._index(mb)
        x, ctx = self._prologue(self.inputs[k], self.seqs[k])
        return self._run(x, ctx), ctx.mod, ctx.head, mb

    def _forward_mid(self, x: Tensor, mod: Tensor, head: Tensor, mb: Tensor):
        seq = self.seqs[self._index(mb)]
        ctx = PartsCtx(seq, mod, self.bands, self.residual, head)
        return self._run(x, ctx), mod, head, mb

    def ctx(self, seq: SeqInfo, mod: Tensor, head: Tensor) -> PartsCtx:
        return PartsCtx(seq, mod, self.bands, self.residual, head)

    def _prologue(self, data: MicroBatch, seq: SeqInfo):
        img = _pad_rows(data.img, seq.n_img)
        text = _pad_rows(data.text, seq.n_txt)
        sigma = _pad_rows(data.sigma, seq.batch)
        return self.parts[0].prologue(img, text, sigma, seq)


def _pad_rows(t: Tensor, rows: int) -> Tensor:
    if t.shape[0] == rows:
        return t
    return torch.cat([t, t.new_zeros(rows - t.shape[0], *t.shape[1:])])


class PipelineEngine:
    """Builds this rank's stages and schedule; `step(microbatches, denom)` runs fwd+bwd."""

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
        self.rank, self.world, self.device, self.shape = rank, world, device, shape
        self.dtype = compute_dtype
        ids = stage_ids(rank, world, schedule, per_rank)
        n_stages = world * (1 if schedule in SINGLE else len(ids))
        if len(bounds) != n_stages:
            raise ValueError(f"need {n_stages} layer ranges, got {len(bounds)}")
        self.modules_: list[StageModule] = []
        stages = []
        for sid in ids:
            lo, hi = bounds[sid]
            mod = StageModule(parts, lo, hi, sid == 0, sid == n_stages - 1, runner)
            mod.to(device)
            self.modules_.append(mod)
            stages.append(self._stage(mod, sid, n_stages, group))
        self.first = any(m.first for m in self.modules_)
        self.last = any(m.last for m in self.modules_)
        self.tail = next((m for m in self.modules_ if m.last), None)
        self.objective = objective
        self.denom = 1.0
        self._losses = 0
        self.targets: list[MicroBatch] = []
        cls = schedule_class(schedule)
        target = stages[0] if schedule in SINGLE else stages
        self.schedule = cls(
            target, shape.microbatches, loss_fn=self._loss, scale_grads=False
        )

    def _example(self, first: bool):
        """Static stage metadata: `(mb,)` into the first stage, else `(x, mod, head,
        mb)`; the float tensors carry gradients."""
        s, d = self.shape, self.device
        mb = torch.zeros(1, dtype=torch.int64, device=d)
        if first:
            return (mb,)
        x = torch.empty(s.tokens, s.dim, dtype=self.dtype, device=d)
        mod = torch.empty(s.max_samples + 1, 6, s.dim, dtype=torch.float32, device=d)
        head = torch.empty(s.max_samples + 1, 2, s.dim, dtype=torch.float32, device=d)
        return (x.requires_grad_(), mod.requires_grad_(), head.requires_grad_(), mb)

    def _stage(self, mod: StageModule, sid: int, n: int, group):
        kw = {}
        if "input_args" in _STAGE_ARGS:
            kw["input_args"] = self._example(mod.first)
        if "output_args" in _STAGE_ARGS:
            kw["output_args"] = self._example(False)
        if "group" in _STAGE_ARGS and group is not None:
            kw["group"] = group
        return pipelining.PipelineStage(mod, sid, n, self.device, **kw)

    def _loss(self, output, target) -> Tensor:
        x, mod, head, mb = output
        k = self._losses
        self._losses = k + 1
        data, tail = self.targets[k], self.tail
        ctx = tail.ctx(tail.seqs[k], mod, head)
        pred = tail.parts[0].epilogue(x, ctx)[: data.target.shape[0]]
        return self.objective(pred, data) / self.denom

    def step(self, microbatches: list[MicroBatch], denom: float) -> Tensor | None:
        """One forward+backward over the step; returns the loss on the last stage."""
        seqs = [padded_seqinfo(mb, self.shape).to(self.device) for mb in microbatches]
        moved = [microbatch_to(mb, self.device) for mb in microbatches]
        for m in self.modules_:
            m.set_step(seqs, moved)
        self.targets, self.denom = moved, denom
        self._losses = 0
        index = torch.arange(len(microbatches), device=self.device)
        args = (index,) if self.first else ()
        losses: list[Tensor] = []
        kwargs = {"target": index, "losses": losses} if self.last else {}
        self.schedule.step(*args, **kwargs)
        if not self.last:
            return None
        return torch.stack([v.detach().float() for v in losses]).sum()

    def parameters(self):
        for m in self.modules_:
            yield from m.parameters()


def microbatch_to(mb: MicroBatch, device) -> MicroBatch:
    """Tensors to `device`; `img_keep` stays on the host (layout input)."""
    moved = {
        f: getattr(mb, f).to(device, non_blocking=True)
        for f in MicroBatch.__dataclass_fields__
        if f != "img_keep"
    }
    return MicroBatch(**moved, img_keep=mb.img_keep)


def broadcast_loss(loss: Tensor | None, device, world: int) -> Tensor:
    """The last stage's loss on every rank."""
    buf = torch.zeros(1, device=device)
    if loss is not None:
        buf[0] = loss
    dist.broadcast(buf, src=world - 1)
    return buf[0]
