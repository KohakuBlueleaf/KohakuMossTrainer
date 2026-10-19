"""MossBackbone: (packed latents, packed text, sigma, SeqInfo) -> packed velocity."""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.model.block import JointBlock
from kohakumoss.model.components.embed import (
    RopeBands,
    TimeEmbedding,
    timestep_embedding,
)
from kohakumoss.model.components.norm import RMSNorm
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import OpSet
from kohakumoss.model.runner import PlainRunner
from kohakumoss.model.stream import StreamConfig, caster, scaler
from kohakumoss.model.text_block import TextEmbedder
from kohakumoss.types.arch import MossArchConfig
from kohakumoss.types.context import HEAD_SCALE, HEAD_SHIFT, BlockCtx
from kohakumoss.types.protocols import BlockRunner
from kohakumoss.types.seqinfo import SeqInfo


class MossBackbone(nn.Module):
    """MOSS-Image DiT. Module names equal the official state-dict keys."""

    def __init__(self, cfg: MossArchConfig, device=None, dtype=None):
        super().__init__()
        cfg.validate()
        kw = dict(device=device, dtype=dtype)
        self.cfg = cfg
        self.video_embedder = MossLinear(cfg.latent_channels, cfg.dim, **kw)
        self.text_embedder = TextEmbedder(cfg, **kw)
        self.rope = RopeBands(cfg.rope_bands, cfg.rope_base, device=device)
        self.time_embedding = TimeEmbedding(cfg.freq_dim, cfg.dim, **kw)
        self.shared_block_modulation = MossLinear(cfg.dim, 6 * cfg.dim, **kw)
        self.layers = nn.ModuleList(JointBlock(cfg, i, **kw) for i in range(cfg.depth))
        self.final_norm_video = RMSNorm(cfg.dim, cfg.eps, **kw)
        self.final_linear_video = MossLinear(
            cfg.dim, cfg.latent_channels, bias=False, **kw
        )
        self.video_head_modulation = MossLinear(cfg.dim, 2 * cfg.dim, **kw)
        self.residual_scale: tuple[float, ...] = (1.0,) * cfg.depth
        self.fold_scales: tuple[float, ...] | None = None
        self.runner: BlockRunner = PlainRunner()
        self.stream = StreamConfig()
        self.fp16_fix = "none"
        self.weight_path: str | None = None
        self._default = self._plan_forward()

    def init_weights(self, std: float = 0.02) -> None:
        """Random init (normal weights, zero biases)."""
        for module in self.modules():
            match module:
                case MossLinear():
                    module.init_weights(std)
                case RMSNorm():
                    module.init_weights()

    def bind_ops(self, ops: OpSet | None = None) -> None:
        """Bind (or re-bind after a structural change) the resolved ops and the stream
        config to every block."""
        self.ops = ops if ops is not None else self.ops
        stream = self.stream
        compute = self.layers[0].proj["q"].weight_like.dtype
        self._cond_in = tuple(m.weight.dtype for m in self.conditioning_linears())
        self._stream_dtype = stream.stream_dtype or compute
        self._video_in = scaler(stream.embed_scale, stream.stream_dtype)
        self._head_cast = caster(compute if stream.stream_dtype is not None else None)
        self.text_embedder.bind(self.ops, stream)
        for i, block in enumerate(self.layers):
            block.bind(self.ops, stream.block(i))

    def set_stream_config(self, stream: StreamConfig) -> None:
        """Install a stream config (fp16 fix) and re-bind."""
        self.stream = stream
        self.bind_ops()

    def set_runner(self, runner: BlockRunner) -> None:
        self.runner = runner

    def _plan(
        self, skip_layers: frozenset, segment_skip: frozenset = frozenset()
    ) -> tuple[Callable, ...]:
        """One step `(model, x, ctx, run) -> x` per block, resolved once: run it;
        (skipped) apply only the fold's stream rescale r (dropped when r == 1);
        (segment-skipped) run it and keep its output only on the tokens `run` marks."""
        steps = []
        for i in range(len(self.layers)):
            r = self.residual_scale[i]
            if i in skip_layers:
                if r != 1.0:
                    steps.append(partial(_rescale, r))
            elif i in segment_skip:
                kept = _same if r == 1.0 else partial(_rescale, r)
                steps.append(partial(_run_selected, i, kept))
            else:
                steps.append(partial(_run_block, i))
        return tuple(steps)

    def _plan_forward(
        self, skip_layers=(), segment_skip_layers=()
    ) -> Callable[..., Tensor]:
        """`bind_forward`'s plan as `(model, img, text, sigma, seq[, mask]) -> v`,
        holding no reference to this model."""
        skip = frozenset(skip_layers)
        if segment_skip_layers:
            segment = frozenset(segment_skip_layers) - skip
            return partial(_forward_masked, self._plan(skip, segment))
        return partial(_forward_plan, self._plan(skip))

    def bind_forward(
        self, skip_layers=(), segment_skip_layers=()
    ) -> Callable[..., Tensor]:
        """A forward with its block plan resolved once (re-bind after a fold change).

        `skip_layers` skip blocks for every sample: `(img, text, sigma, seq) -> v`.
        `segment_skip_layers` skip blocks only for the samples flagged by a bool [B]
        mask: `(img, text, sigma, seq, mask) -> v` (packed CFG); a flagged sample's
        stream takes the skip's value, whatever the block computed on it."""
        return partial(self._plan_forward(skip_layers, segment_skip_layers), self)

    def set_residual_scale(self, scales: tuple[float, ...]) -> None:
        self.residual_scale = tuple(scales)
        self._default = self._plan_forward()

    def conditioning_linears(self) -> tuple[MossLinear, ...]:
        """The KEEP_FP32 linears of the per-sample conditioning."""
        te = self.time_embedding
        return (te[0], te[2], self.shared_block_modulation, self.video_head_modulation)

    def conditioning(self, sigma: Tensor) -> tuple[Tensor, Tensor]:
        """Per-sample fp32 modulation [B, 6, D] and head modulation [B, 2, D]. Each
        linear takes its input in its weight dtype and runs under the caller's autocast,
        like any PyTorch module."""
        d0, d2, dm, dh = self._cond_in
        te = self.time_embedding
        emb = timestep_embedding(sigma.float() * self.cfg.time_scale, self.cfg.freq_dim)
        hidden = te[1](te[0](emb.to(d0)))
        time = te[2](hidden.to(d2)).float()
        mod = self.shared_block_modulation(time.to(dm)).float()
        head = self.video_head_modulation(time.to(dh)).float()
        return mod.unflatten(-1, (6, -1)), head.unflatten(-1, (2, -1))

    def embed(self, img: Tensor, text: Tensor, seq: SeqInfo) -> Tensor:
        """Joint residual stream [N_joint, D] with sample-major [img_i ; txt_i] order."""
        video = self._video_in(
            self.video_embedder(img.to(self.video_embedder.weight.dtype))
        )
        words = self.text_embedder(text, seq, self.rope.bands)
        x = video.new_empty(seq.n_joint, video.shape[-1], dtype=self._stream_dtype)
        x = x.index_copy(0, seq.img_in_joint, video.to(x.dtype))
        return x.index_copy(0, seq.txt_in_joint, words.to(x.dtype))

    def head(self, x: Tensor, head: Tensor, seq: SeqInfo) -> Tensor:
        """Final modulated norm + linear on the image tokens -> [N_img, C]."""
        xi = x.index_select(0, seq.img_in_joint)
        norm = self.final_norm_video
        y = self.ops.rms_norm_mod(
            xi,
            norm.weight,
            head[:, HEAD_SHIFT],
            head[:, HEAD_SCALE],
            seq.seg_img,
            norm.eps,
        )
        return self.final_linear_video(self._head_cast(y))

    def _trunk(self, plan, x: Tensor, ctx: BlockCtx, run) -> Tensor:
        for step in plan:
            x = step(self, x, ctx, run)
        return x

    def forward(self, img: Tensor, text: Tensor, sigma: Tensor, seq: SeqInfo) -> Tensor:
        """img [N_img, C], text [N_txt, text_dim], sigma f32 [B] in [0, 1] -> v [N_img, C].

        Every block runs; `bind_forward` gives skipping variants."""
        return self._default(self, img, text, sigma, seq)


def _forward_plan(plan, model: MossBackbone, img, text, sigma, seq) -> Tensor:
    mod, head = model.conditioning(sigma)
    x = model.embed(img, text, seq)
    ctx = BlockCtx(seq, mod, model.rope.bands, model.residual_scale)
    x = model._trunk(plan, x, ctx, None)
    return model.head(x, head, seq)


def _forward_masked(plan, model: MossBackbone, img, text, sigma, seq, mask) -> Tensor:
    mod, head = model.conditioning(sigma)
    x = model.embed(img, text, seq)
    ctx = BlockCtx(seq, mod, model.rope.bands, model.residual_scale)
    x = model._trunk(plan, x, ctx, runs_on(mask, seq))
    return model.head(x, head, seq)


def _run_block(
    index: int, model: MossBackbone, x: Tensor, ctx: BlockCtx, run
) -> Tensor:
    return model.runner.run(index, model.layers[index], x, ctx)


def _run_selected(index: int, kept, model, x: Tensor, ctx: BlockCtx, run) -> Tensor:
    out = model.runner.run(index, model.layers[index], x, ctx)
    return torch.where(run, out, kept(model, x, ctx, run))


def _rescale(r: float, model, x: Tensor, ctx, run) -> Tensor:
    return x * r


def _same(model, x: Tensor, ctx, run) -> Tensor:
    return x


def runs_on(mask: Tensor, seq: SeqInfo) -> Tensor:
    """Per joint token [N_joint, 1] bool: True where the sample is not flagged by the
    per-sample `mask` [B] (a segment-skipped block's output is kept there)."""
    return (~mask).index_select(0, seq.seg_joint.long()).unsqueeze(-1)


def compute_dtype(model: MossBackbone) -> torch.dtype:
    """Dtype of the joint-block weights (the activation dtype of the trunk)."""
    return model.layers[0].proj["q"].weight_like.dtype
