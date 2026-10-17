"""MossBackbone: (packed latents, packed text, sigma, SeqInfo) -> packed velocity."""

from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.model.block import JointBlock
from kohakumoss.model.components.embed import RopeBands, Sinusoid, TimeEmbedding
from kohakumoss.model.components.norm import RMSNorm
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import OpSet
from kohakumoss.model.runner import PlainRunner
from kohakumoss.model.stream import StreamPolicy, caster, scaler
from kohakumoss.model.text_block import TextEmbedder
from kohakumoss.types.api import BlockCtx
from kohakumoss.types.arch import MossArchConfig
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
        self.stream = StreamPolicy()
        self.fp16_fix = "none"
        self.cond_dtype: torch.dtype | None = None
        self._sinusoid = Sinusoid(cfg.freq_dim, cfg.time_scale)
        self._default = self.bind_forward()

    def init_weights(self, std: float = 0.02) -> None:
        """Random init for from-scratch / tiny models (normal weights, zero biases)."""
        for module in self.modules():
            match module:
                case MossLinear():
                    module.init_weights(std)
                case RMSNorm():
                    module.init_weights()

    def bind_ops(self, ops: OpSet | None = None) -> None:
        """Bind (or re-bind after a structural change) the resolved ops and the stream
        policy to every block."""
        self.ops = ops if ops is not None else self.ops
        policy = self.stream
        compute = self.layers[0].proj["q"].weight.dtype
        device = self.time_embedding[0].weight.device
        self._fp32_region = partial(torch.autocast, device.type, enabled=False)
        wide = self.cond_dtype or compute
        wide = wide if wide in (torch.float16, torch.bfloat16) else None
        for module in self.conditioning_linears():
            module.set_input_dtype(wide)
        self._cond_in = tuple(
            wide or m.weight.dtype for m in self.conditioning_linears()
        )
        self._stream_dtype = policy.stream_dtype or compute
        self._video_in = scaler(policy.embed_scale, policy.stream_dtype)
        self._head_cast = caster(compute if policy.stream_dtype is not None else None)
        self.text_embedder.bind(self.ops, policy)
        for i, block in enumerate(self.layers):
            block.bind(self.ops, policy.block(i))

    def set_stream_policy(self, policy: StreamPolicy) -> None:
        """Install a stream policy (fp16 fix) and re-bind."""
        self.stream = policy
        self.bind_ops()

    def set_runner(self, runner: BlockRunner) -> None:
        self.runner = runner

    def _plan(self, skip_layers: frozenset) -> tuple[Callable, ...]:
        """One step per block, resolved once: run it, or (skipped) apply only the fold's
        stream rescale r (dropped when r == 1)."""
        steps = []
        for i in range(len(self.layers)):
            r = self.residual_scale[i]
            if i not in skip_layers:
                steps.append(partial(self._run_block, i))
            elif r != 1.0:
                steps.append(partial(_rescale, r))
        return tuple(steps)

    def _run_block(self, index: int, x: Tensor, ctxs: tuple[BlockCtx, ...]) -> Tensor:
        return self.runner.run(index, self.layers[index], x, ctxs[index])

    def bind_forward(
        self, skip_layers=(), segment_skip_layers=()
    ) -> Callable[..., Tensor]:
        """A forward with its block plan resolved once (re-bind after a fold change).

        `skip_layers` skip blocks for every sample: `(img, text, sigma, seq) -> v`.
        `segment_skip_layers` skip blocks only for the samples flagged by a bool [B]
        mask: `(img, text, sigma, seq, mask) -> v` (packed CFG)."""
        plan = self._plan(frozenset(skip_layers))
        if segment_skip_layers:
            return partial(self._forward_masked, plan, tuple(segment_skip_layers))
        return partial(self._forward_plan, plan)

    def set_residual_scale(self, scales: tuple[float, ...]) -> None:
        self.residual_scale = tuple(scales)
        self._default = self.bind_forward()

    def conditioning_linears(self) -> tuple[MossLinear, ...]:
        """The KEEP_FP32 linears of the per-sample conditioning."""
        te = self.time_embedding
        return (te[0], te[2], self.shared_block_modulation, self.video_head_modulation)

    def conditioning(self, sigma: Tensor) -> tuple[Tensor, Tensor]:
        """Per-sample fp32 modulation [B, 6, D] and head modulation [B, 2, D], with any
        enclosing autocast disabled. With a 16-bit trunk (or `cond_dtype`) the linears
        take 16-bit inputs and accumulate and return fp32; otherwise they run in their
        own dtype."""
        d0, d2, dm, dh = self._cond_in
        te = self.time_embedding
        with self._fp32_region():
            emb = self._sinusoid(sigma.float() * self.cfg.time_scale)
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
            xi, norm.weight, head[:, 0], head[:, 1], seq.seg_img, norm.eps
        )
        return self.final_linear_video(self._head_cast(y))

    def _trunk(self, plan, x: Tensor, ctxs: tuple[BlockCtx, ...]) -> Tensor:
        for step in plan:
            x = step(x, ctxs)
        return x

    def _forward_plan(self, plan, img, text, sigma, seq) -> Tensor:
        mod, head = self.conditioning(sigma)
        x = self.embed(img, text, seq)
        ctx = BlockCtx(seq, mod, self.rope.bands, self.residual_scale)
        x = self._trunk(plan, x, (ctx,) * len(self.layers))
        return self.head(x, head, seq)

    def _forward_masked(self, plan, layers, img, text, sigma, seq, mask) -> Tensor:
        mod, head = self.conditioning(sigma)
        x = self.embed(img, text, seq)
        ctx = BlockCtx(seq, mod, self.rope.bands, self.residual_scale)
        ctxs = [ctx] * len(self.layers)
        masked = _gates_zeroed(ctx, mask)
        for i in layers:
            ctxs[i] = masked
        x = self._trunk(plan, x, tuple(ctxs))
        return self.head(x, head, seq)

    def forward(self, img: Tensor, text: Tensor, sigma: Tensor, seq: SeqInfo) -> Tensor:
        """img [N_img, C], text [N_txt, text_dim], sigma f32 [B] in [0, 1] -> v [N_img, C].

        Every block runs; `bind_forward` gives skipping variants."""
        return self._default(img, text, sigma, seq)


def _rescale(r: float, x: Tensor, ctxs) -> Tensor:
    return x * r


def _gates_zeroed(ctx: BlockCtx, mask: Tensor) -> BlockCtx:
    """Context whose gates are zero for masked samples: a block then leaves their stream
    untouched, exactly as skipping it would."""
    keep = (~mask).to(ctx.mod.dtype)[:, None]
    mod = ctx.mod.clone()
    mod[:, 2] = mod[:, 2] * keep
    mod[:, 5] = mod[:, 5] * keep
    return BlockCtx(ctx.seq, mod, ctx.bands, ctx.residual_scale)


def all_linears(model: nn.Module) -> dict[str, MossLinear]:
    """Every MossLinear by dotted path."""
    return {n: m for n, m in model.named_modules() if isinstance(m, MossLinear)}


def compute_dtype(model: MossBackbone) -> torch.dtype:
    """Dtype of the joint-block weights (the activation dtype of the trunk)."""
    return model.layers[0].proj["q"].weight.dtype
