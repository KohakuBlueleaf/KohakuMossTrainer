"""Gradient references and kernel sweeps for scripts/probe/train_batch.py.

A reference re-runs one prepared batch through a fresh copy of the model (CPU fp32 /
fp64, or GPU fp32 compute over the bf16 checkpoint weights) with the live adapter
values, and returns the flat adapter gradient for comparison with the bf16 run.
"""

import json
import math
import re
from collections import defaultdict
from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as F
from official_lora import OfficialLoRA

from kohakumoss.adapters.apply import apply as apply_adapters
from kohakumoss.adapters.spec import spec_from_knobs
from kohakumoss.model import load_backbone
from kohakumoss.model.linear import MossLinear
from kohakumoss.model.opset import FUSED, PRIMITIVES, OpSet
from kohakumoss.training.memory.runners import CheckpointRunner
from kohakumoss.training.step import image_tokens_per_sample

REF_DTYPES = {"fp32": torch.float32, "fp64": torch.float64}
EAGER = {name: "eager" for name in PRIMITIVES + FUSED}


def split(t: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    return list(t.split(sizes, 0))


def sample_mse(pred, target, sizes) -> list[float]:
    err = (pred.float() - target.float()).pow(2).mean(-1)
    return [float(e.mean()) for e in split(err, sizes)]


def group_of(name: str) -> str:
    """Parameter name -> (layer kind) group with digits folded."""
    return re.sub(r"\d+", "#", name)


def layer_of(name: str) -> str:
    m = re.search(r"layers[._](\d+)", name)
    return f"layer{m.group(1)}" if m else "other"


def grad_report(trainable) -> dict:
    """Finite gradient norm (total, per group, per layer) and the non-finite params."""
    by_group = defaultdict(float)
    by_layer = defaultdict(float)
    bad = []
    bad_layers = defaultdict(int)
    none = 0
    total = 0.0
    for name, p in trainable.named_trainable():
        if p.grad is None:
            none += 1
            continue
        g = p.grad.detach().double()
        if not torch.isfinite(g).all():
            bad.append({"name": name, "nonfinite": int((~torch.isfinite(g)).sum())})
            bad_layers[layer_of(name)] += 1
            continue
        sq = float(g.pow(2).sum())
        by_group[group_of(name)] += sq
        by_layer[layer_of(name)] += sq
        total += sq
    return {
        "finite_norm": math.sqrt(total),
        "no_grad": none,
        "nonfinite_params": len(bad),
        "nonfinite": bad[:64],
        "nonfinite_by_layer": dict(bad_layers),
        "by_group": {k: math.sqrt(v) for k, v in sorted(by_group.items())},
        "by_layer": {k: math.sqrt(v) for k, v in sorted(by_layer.items())},
    }


def flat_grad(trainable) -> torch.Tensor:
    return torch.cat(
        [
            q.grad.detach().flatten().double().cpu()
            for _, q in trainable.named_trainable()
        ]
    )


def compare(flat: torch.Tensor, ref: torch.Tensor, tag: str) -> dict:
    """Relative L2 error and cosine of a flat gradient against a reference."""
    flat, ref = flat.double(), ref.double()
    return {
        f"grad_rel_vs_{tag}": float((flat - ref).norm() / ref.norm()),
        f"grad_cos_vs_{tag}": float((flat @ ref) / (flat.norm() * ref.norm())),
    }


def layer_slices(trainable) -> dict[str, slice]:
    """Flat-gradient index range of each layer's adapter parameters."""
    out, start = {}, 0
    for name, q in trainable.named_trainable():
        layer = layer_of(name)
        lo, _ = out.get(layer, (start, start))
        out[layer] = (lo, start + q.numel())
        start += q.numel()
    return {k: slice(*v) for k, v in out.items()}


def variant_row(step, autocast, p, denom, img_sizes, bundle, trainable):
    """Packed forward + backward of prepared batch `p` under the currently bound ops:
    (summary row, flat fp64 gradient on CPU)."""
    bundle.zero_grad()
    with autocast():
        pred = step.backbone(p.z, p.text, p.sigma, p.seq)
    loss = step.loss(pred, p, denom)
    loss.backward()
    grads = grad_report(trainable)
    flat = flat_grad(trainable)
    bundle.zero_grad()
    return {
        "mse_per_sample": sample_mse(pred.detach(), p.target, img_sizes),
        "grad_nonfinite_params": grads["nonfinite_params"],
        "grad_finite_norm": grads["finite_norm"],
        "grad_by_layer": grads["by_layer"],
    }, flat


class RefModel:
    """A fresh model in compute `dtype` on `device` with torch adapter bodies holding
    `live`'s adapter values. `block_linear` stores the joint-block linear weights in
    that dtype (the bf16 checkpoint values, exact) while every activation stays in
    `dtype`."""

    def __init__(self, k, live, device, dtype, block_linear=None):
        model, _ = load_backbone(
            k["WEIGHTS"], device="cpu", dtype=dtype, keep_fp32=(), kernels={}
        )
        model.requires_grad_(False)
        if block_linear is not None:
            for module in model.layers.modules():
                if isinstance(module, MossLinear):
                    module.weight.data = module.weight.data.to(block_linear)
        model.to(device)
        spec = spec_from_knobs({**k, "KERNELS": {"lycoris": "torch"}})
        self.aset = apply_adapters(model, spec, dtype=dtype, device=device)
        values = dict(live.named_trainable())
        with torch.no_grad():
            for name, q in self.aset.named_trainable():
                q.copy_(values[name].detach().to(device, dtype))
        self.model, self.device, self.dtype = model, device, dtype

    def grad(self, p, denom, kernels, emulate=frozenset()) -> torch.Tensor:
        """Flat adapter gradient of prepared batch `p` under ops `kernels` with the bf16
        roundings named in `emulate` (see `emulated_ops`, `StreamRounding`; "inputs"
        rounds the noised latents to bf16, as the bf16 embedder input does)."""
        model, device, dtype = self.model, self.device, self.dtype
        ops = emulated_ops(OpSet.resolve(device, dtype, kernels), emulate)
        model.set_runner(
            StreamRounding(CheckpointRunner("block").bind(model.layers), emulate)
        )
        model.bind_ops(ops)
        model._stream_dtype = dtype
        for q in self.aset.trainable_params():
            q.grad = None
        seq = p.seq.to(device)
        z, text = p.z.to(device, dtype), p.text.to(device, dtype)
        if "inputs" in emulate:
            z = z.to(torch.bfloat16).to(dtype)
        with bf16_linears(model, "linears" in emulate):
            pred = model(z, text, p.sigma.to(device), seq)
            err = (pred - p.target.to(device, dtype)).square().mean(-1)
            weight = p.weight.to(device, dtype).index_select(0, seq.seg_img.long())
            ((err * weight).sum() / float(denom)).backward()
        return flat_grad(self.aset)

    def close(self) -> None:
        self.model.to("meta")
        del self.model, self.aset
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


ROUND_ARGS = {
    "rms_norm": (0, 1),
    "rms_norm_mod": (0, 1),
    "qk_norm_rope": (0, 1, 2, 3),
    "attention_varlen": (0, 1, 2),
    "situ_glu": (0,),
    "gated_residual": (0, 1),
}


def _rounded(fn, idx, *args):
    """`fn` on args `idx` rounded to bf16 (as the bf16 path holds them); outputs and
    incoming gradients pass through bf16."""
    a = [
        x.to(torch.bfloat16) if i in idx and x is not None else x
        for i, x in enumerate(args)
    ]
    out = fn(*a)
    if isinstance(out, tuple):
        return tuple(o.float() for o in out)
    return out.float()


def emulated_ops(ops: OpSet, emulate) -> OpSet:
    """`ops` with each primitive named in `emulate` run on bf16-rounded inputs."""
    fns = {
        name: partial(_rounded, getattr(ops, name), ROUND_ARGS[name])
        for name in ROUND_ARGS
        if name in emulate
    }
    return replace(ops, fused={}, **fns)


class _RoundValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.to(torch.bfloat16).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


class _RoundGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.to(torch.bfloat16).to(g.dtype)


class StreamRounding:
    """Runner wrapper: the residual stream between blocks rounded to bf16 in value
    ("stream_fwd") and / or in gradient ("stream_bwd")."""

    def __init__(self, inner, emulate):
        self.inner = inner
        self.fwd = "stream_fwd" in emulate
        self.bwd = "stream_bwd" in emulate

    def bind(self, layers):
        self.inner.bind(layers)
        return self

    def run(self, index, block, x, ctx):
        y = self.inner.run(index, block, x, ctx)
        y = _RoundValue.apply(y) if self.fwd else y
        return _RoundGrad.apply(y) if self.bwd else y


def _bf16_linear(self, x):
    b = None if self.bias is None else self.bias.to(torch.bfloat16)
    w = self.weight.to(torch.bfloat16)
    return F.linear(x.to(torch.bfloat16), w, b).to(x.dtype)


class bf16_linears:
    """Context: every dense MossLinear of `model` computes in bf16 (inputs, weight,
    output) when `on`."""

    def __init__(self, model, on: bool):
        self.linears = [m for m in model.modules() if isinstance(m, MossLinear)]
        self.on = on
        self.saved = (MossLinear._dense, MossLinear._dense_nobias)

    def _rebind(self) -> None:
        for m in self.linears:
            m.rebind()

    def __enter__(self):
        if self.on:
            MossLinear._dense = MossLinear._dense_nobias = _bf16_linear
            self._rebind()
        return self

    def __exit__(self, *exc):
        MossLinear._dense, MossLinear._dense_nobias = self.saved
        self._rebind()
        return False


def bf16_variants(per_op: bool) -> dict[str, dict]:
    """Kernel choices of the bf16 sweep: all-eager, default, and (per_op) each primitive
    op flipped alone in both directions."""
    variants = {
        "eager": EAGER,
        "eager_again": EAGER,
        "default": {},
        "default_again": {},
    }
    for name in PRIMITIVES if per_op else ():
        variants[f"eager_only:{name}"] = {name: "eager"}
        flip = (name,) if name == "rms_norm_mod" else (name, f"linear_{name}")
        variants[f"default_only:{name}"] = {
            key: v for key, v in EAGER.items() if key not in flip
        }
    return variants


def report_against(rows, flats, slices, tag) -> None:
    """Add whole-vector and per-layer agreement with reference `tag` to every row."""
    ref = flats[tag]
    for label, flat in flats.items():
        rows.setdefault(label, {}).update(compare(flat, ref, tag))
        rows[label][f"layer_cos_vs_{tag}"] = {
            name: compare(flat[s], ref[s], tag)[f"grad_cos_vs_{tag}"]
            for name, s in slices.items()
        }
        brief = {
            key: v
            for key, v in rows[label].items()
            if key.startswith("grad_") and key.endswith(f"_{tag}")
        }
        print(tag, label, json.dumps(brief), flush=True)


def op_sweep(
    k, built, step, batcher, autocast, bundle, hb, device, dtype, opts
) -> dict:
    """bf16 sweep of `bf16_variants`, then the GPU fp32 references ("gpu_fp32": eager /
    default kernels), the bf16 emulations ("emulate") and the CPU references ("cpu":
    fp32 / fp64) of the same prepared batch; every row compared with every reference.
    With `draws` > 1 each gradient is the mean over that many sigma / noise draws of the
    batch (label) next to its first draw alone (label@0), and `_curves` holds the
    agreement of the first-n means for n = 1, 2, 4, ...; "official" adds the official
    transformer in bf16 autocast with the same LoRA (official_lora.py)."""
    batch = batcher(hb)
    img_sizes = image_tokens_per_sample(hb)
    ps = [step.prepare(batch, 0) for _ in range(opts["draws"])]
    denom = step.denominator(batch)
    backbone = built.backbone
    rows, flats, per_draws = {}, {}, {}

    def keep(label, per_draw):
        per_draws[label] = [f.float() for f in per_draw]
        flats[f"{label}@0"] = per_draw[0]
        flats[label] = torch.stack(per_draws[label]).double().mean(0)

    for label, choices in bf16_variants(opts["per_op"]).items():
        backbone.bind_ops(OpSet.resolve(device, dtype, choices))
        runs = [
            variant_row(step, autocast, p, denom, img_sizes, bundle, built.trainable)
            for p in ps
        ]
        keep(label, [flat for _, flat in runs])
        rows[label] = {**runs[0][0], **compare(flats[label], flats["eager"], "eager")}
        brief = {key: v for key, v in rows[label].items() if key != "grad_by_layer"}
        print(label, json.dumps(brief), flush=True)
    backbone.bind_ops(OpSet.resolve(device, dtype, {}))
    slices = layer_slices(built.trainable)
    tags = []
    if opts["gpu_fp32"] or opts["emulate"]:
        backbone.to("cpu")
        torch.cuda.empty_cache()
        ref = RefModel(k, built.trainable, device, torch.float32, torch.bfloat16)
        for name in opts["gpu_fp32"]:
            kernels = EAGER if name == "eager" else {}
            keep(f"gpu_fp32_{name}", [ref.grad(p, denom, kernels) for p in ps])
            print("done", f"gpu_fp32_{name}", flush=True)
        for spec in opts["emulate"]:
            emulate = frozenset(spec.split("+"))
            keep(f"emulate:{spec}", [ref.grad(p, denom, EAGER, emulate) for p in ps])
            print("done", f"emulate:{spec}", flush=True)
        ref.close()
        tags += [f"gpu_fp32_{name}" for name in opts["gpu_fp32"]]
        tags += [f"gpu_fp32_{name}@0" for name in opts["gpu_fp32"]]
    if opts["official"]:
        backbone.to("cpu")
        torch.cuda.empty_cache()
        official = OfficialLoRA(k["WEIGHTS"], built.trainable, device)
        keep("official_bf16", [official.grad(p, denom) for p in ps])
        official.close()
        print("done official_bf16", flush=True)
    backbone.to(device)
    for tag in opts["cpu"]:
        ref = RefModel(k, built.trainable, torch.device("cpu"), REF_DTYPES[tag])
        keep(f"cpu_{tag}", [ref.grad(p, denom, {}) for p in ps])
        ref.close()
        tags.append(f"cpu_{tag}")
    for tag in tags:
        report_against(rows, flats, slices, tag)
    if tags and opts["draws"] > 1:
        rows["_curves"] = mean_curves(per_draws, tags[0], slices)
    return rows


def mean_curves(per_draws: dict, tag: str, slices: dict) -> dict:
    """cos(mean of the first n draws of each label, same mean of reference `tag`) for n
    = 1, 2, 4, ..., overall and per layer."""
    ref = per_draws[tag]
    total = len(ref)
    ns = [n for n in (2**i for i in range(total.bit_length())) if n <= total]
    out = {}
    for label, draws in per_draws.items():
        rows = {}
        for n in ns:
            a = torch.stack(draws[:n]).double().mean(0)
            b = torch.stack(ref[:n]).double().mean(0)
            rows[n] = {
                "cos": compare(a, b, tag)[f"grad_cos_vs_{tag}"],
                "layers": {
                    name: compare(a[s], b[s], tag)[f"grad_cos_vs_{tag}"]
                    for name, s in slices.items()
                },
            }
        per_draw = [
            (
                compare(a.double(), b.double(), tag)[f"grad_cos_vs_{tag}"],
                float(a.double().norm() / b.double().norm()),
            )
            for a, b in zip(draws, ref)
        ]
        out[label] = {"means": rows, "per_draw_cos_norm_ratio": per_draw}
        print(
            "curve", label, {n: round(r["cos"], 4) for n, r in rows.items()}, flush=True
        )
        print("draws", label, [(round(c, 3), round(r, 2)) for c, r in per_draw])
    return out


class AttentionDump:
    """Bound as one block's `attention_varlen`: records each autograd call's (q, k, v,
    cu, max_seqlen, scale) and the gradient arriving at its output; `saved` is the
    first record that received a gradient."""

    def __init__(self, inner):
        self.inner = inner
        self.records = []

    def __call__(self, q, k, v, cu, max_seqlen, scale):
        out = self.inner(q, k, v, cu, max_seqlen, scale)
        if out.requires_grad:
            rec = {
                "q": q.detach().clone(),
                "k": k.detach().clone(),
                "v": v.detach().clone(),
                "cu": cu.detach().clone(),
                "max_seqlen": int(max_seqlen),
                "scale": float(scale),
            }
            self.records.append(rec)
            out.register_hook(partial(_keep_grad, rec))
        return out

    @property
    def saved(self) -> dict:
        return next((r for r in self.records if "grad_out" in r), {})


def _keep_grad(rec: dict, grad):
    rec["grad_out"] = grad.detach().clone()


def dump_attention(backbone, layer: int) -> AttentionDump:
    """Route block `layer`'s attention through an AttentionDump."""
    block = backbone.layers[layer]
    dump = AttentionDump(block.ops.attention_varlen)
    block.ops = replace(block.ops, attention_varlen=dump)
    return dump


def replay(step, dataset, start: int, stop: int, cfg, device) -> int:
    """Draw the sigma and noise of batches [start, stop) as TrainStep.prepare would,
    without running them; returns `stop`."""
    for index in range(start, stop):
        hb = dataset.materialize(index)
        step.sample_sigma(len(hb.ids), step.sigma_gen, index)
        n = sum(image_tokens_per_sample(hb))
        torch.randn(
            (n, cfg.latent_channels),
            generator=step.noise_gen,
            device=device,
            dtype=torch.float32,
        )
    return stop
