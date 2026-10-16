"""LycoAdapterSet: the AdapterSet over LyCORIS modules bound to a model."""

import json
from dataclasses import asdict
from functools import partial

import torch
from lycoris.modules.tlora import clear_timestep_mask
from torch import Tensor, nn

from kohakumoss.adapters import norms as norm_delta
from kohakumoss.adapters.bind.select import bind_target, training_dependent
from kohakumoss.adapters.delta import delta
from kohakumoss.adapters.export import convert, files
from kohakumoss.adapters.export.merged import merged_state_dict
from kohakumoss.adapters.spec import AdapterSpec
from kohakumoss.adapters.target import CHILD, NormTarget, Target

NO_DECAY = (
    "dora_scale",
    "rescale",
    "scalar",
    "lambda_layer",
    "alpha",
)
UP_SIDE = ("lora_up", "hada_w1_a", "hada_w2_a", "lokr_w2_b", "b1", "p_layer", "up_list")
FORMATS = ("lycoris", "comfy", "merged", "diffusers")
RAW_TAG = "kohakumoss_adapter_state"


def add_bias(linear: nn.Module, value: Tensor) -> None:
    """Set the linear's bias to `value`, creating it on a bias-less layer (and
    re-selecting MossLinear's bias / no-bias forward)."""
    if linear.bias is None:
        dtype = linear.weight.dtype if linear.weight.numel() else torch.float32
        param = torch.zeros(value.shape, device=value.device, dtype=dtype)
        linear.bias = nn.Parameter(param, requires_grad=False)
    linear.bias.data.copy_(value.to(linear.bias))
    getattr(linear, "rebind", _noop)()


def _noop() -> None:
    return None


def _train_and_rebind(aset, t: Target, mode: bool = True):
    """Module.train for a LyCORIS module whose bound forward depends on the mode."""
    nn.Module.train(t.lyco, mode)
    aset.bind_one(t)
    return t.lyco


class LycoAdapterSet:
    """Adapters on `model`; forwards bound once, rebound on multiplier changes."""

    def __init__(
        self,
        model: nn.Module,
        spec: AdapterSpec,
        targets: list[Target],
        norms: list[NormTarget],
        backend: str,
    ):
        self.model = model
        self.spec = spec
        self.targets = targets
        self.norms = norms
        self.backend = backend
        self.multiplier = spec.multiplier
        self.by_name = {t.lora_name: t for t in targets}
        self.bind()

    def bind(self) -> None:
        """Install every layer's forward for the current multiplier and training state;
        layers whose forward depends on train()/eval() re-bind through a hook."""
        for t in self.targets:
            self.bind_one(t)
            if training_dependent(t, self.spec.module_dropout):
                vars(t.lyco)["train"] = partial(_train_and_rebind, self, t)
        for n in self.norms:
            for _, d in n.deltas():
                d.multiplier = self.multiplier

    def bind_one(self, t: Target) -> None:
        fn = bind_target(
            t,
            self.spec.mode,
            self.backend,
            self.multiplier,
            self.spec.dora_detach,
            self.spec.module_dropout,
            race=self.spec.backend == "auto",
        )
        if t.kind != "full":
            t.linear.forward = fn

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)
        self.bind()

    @staticmethod
    def _norm_params(n: NormTarget):
        return [(name, d.w_norm) for name, d in n.deltas()]

    def trainable_params(self):
        for name, p in self.named_trainable():
            yield p

    def named_trainable(self):
        for t in self.targets:
            for name, p in t.lyco.named_parameters():
                if p.requires_grad:
                    yield f"{t.lora_name}.{name}", p
        for n in self.norms:
            for name, p in self._norm_params(n):
                if p.requires_grad:
                    yield f"{n.lora_name}.{name}", p

    def param_groups(self, lr: float, weight_decay: float = 0.0, **rules) -> list[dict]:
        """Groups: decayed / not decayed, and the LoRA+ up side at lr * ratio."""
        ratio = float(rules.get("lr_ratio_up", self.spec.lr_ratio_up))
        buckets: dict[tuple[bool, bool], list] = {}
        for name, p in self.named_trainable():
            tail = name.split(".", 1)[1]
            no_decay = p.dim() < 2 or any(tag in tail for tag in NO_DECAY)
            up = ratio != 1.0 and any(tail.startswith(tag) for tag in UP_SIDE)
            buckets.setdefault((no_decay, up), []).append(p)
        groups = []
        for (no_decay, up), params in sorted(buckets.items()):
            groups.append(
                {
                    "params": params,
                    "lr": lr * (ratio if up else 1.0),
                    "weight_decay": 0.0 if no_decay else weight_decay,
                }
            )
        return groups

    def state_dict(self) -> dict[str, Tensor]:
        """Raw trainable state (exact resume): every parameter and buffer by name."""
        out = {}
        for t in self.targets:
            for name, v in nn.Module.state_dict(t.lyco).items():
                out[f"{t.lora_name}.{name}"] = v.detach().clone()
        for n in self.norms:
            for name, v in self._norm_params(n):
                out[f"{n.lora_name}.{name}"] = v.detach().clone()
        return out

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Copy a `state_dict()` back in place; every key must match a live tensor."""
        live = {}
        for t in self.targets:
            for name, v in list(t.lyco.named_parameters()) + list(
                t.lyco.named_buffers()
            ):
                live[f"{t.lora_name}.{name}"] = v
        for n in self.norms:
            for name, v in self._norm_params(n):
                live[f"{n.lora_name}.{name}"] = v
        missing = [k for k in state if k not in live]
        if missing:
            raise KeyError(f"unknown adapter keys: {missing[:8]} ...")
        with torch.no_grad():
            for key, value in state.items():
                live[key].copy_(value.to(live[key].device, live[key].dtype))
        self.bind()

    def tensors(self, fmt: str, style: str = "lora_unet", row_scales=None) -> dict:
        """Adapter tensors in `fmt` ('lycoris' | 'comfy') under key `style`.

        `row_scales` is the residual fold's per-output-row scale of each folded layer
        (model.numerics.output_row_scales); ΔW rows are divided by it.
        """
        clear_timestep_mask(0)
        out = {}
        with torch.no_grad():
            for t in self.targets:
                per = convert.native(t) if fmt == "lycoris" else convert.comfy(t)
                if row_scales and t.path in row_scales:
                    inv = convert.reciprocal(row_scales[t.path])
                    per = convert.apply_row_scale(t, per, inv)
                stem = (
                    t.lora_name if fmt == "lycoris" else files.layer_key(t.path, style)
                )
                out.update({f"{stem}.{k}": v for k, v in per.items()})
            for n in self.norms:
                stem = (
                    n.lora_name if fmt == "lycoris" else files.layer_key(n.path, style)
                )
                out.update(
                    {f"{stem}.{k}": v for k, v in convert.norm_native(n).items()}
                )
        return out

    def metadata(
        self, fmt: str, style: str, extra: dict | None = None
    ) -> dict[str, str]:
        meta = {
            "kohakumoss_format": fmt,
            "kohakumoss_key_style": style,
            "kohakumoss_prefix": self.spec.prefix,
            "kohakumoss_algo": self.spec.algo,
            "ss_network_module": "lycoris.kohya",
            "ss_network_dim": str(self.spec.dim),
            "ss_network_alpha": str(self.spec.alpha),
            "ss_network_args": json.dumps(
                {"algo": self.spec.lycoris_algo, **self.spec.module_kwargs()},
                default=str,
            ),
        }
        meta.update({k: str(v) for k, v in (extra or {}).items()})
        return meta

    def export(
        self,
        path: str,
        fmt: str,
        metadata: dict[str, str] | None = None,
        style: str = "lora_unet",
        dtype: torch.dtype | None = torch.bfloat16,
        row_scales: dict | None = None,
        fused_qkv: bool = False,
        config: dict | None = None,
    ) -> None:
        """Write the adapters (lycoris / comfy) or a merged checkpoint (merged / diffusers).

        `row_scales` (from a folded model) is undone so files are in unfolded space.
        """
        if fmt not in FORMATS:
            raise KeyError(f"export format {fmt!r} not in {FORMATS}")
        if fmt in ("lycoris", "comfy"):
            tensors = files.to_storage(self.tensors(fmt, style, row_scales), dtype)
            files.write(path, tensors, self.metadata(fmt, style, metadata))
            return
        inverse = {k: convert.reciprocal(v) for k, v in (row_scales or {}).items()}
        state = merged_state_dict(
            self.model,
            self.targets,
            self.norms,
            self.multiplier,
            dtype,
            inverse,
            fused_qkv,
        )
        state = files.to_storage(state, None)
        if fmt == "merged":
            files.write(
                path, state, {"kohakumoss_format": "merged", **(metadata or {})}
            )
        else:
            files.write_sharded(path, state, config)

    def save_state(self, path: str) -> None:
        """Raw state + the spec as a safetensors file (checkpoints, exact reload)."""
        meta = {RAW_TAG: "1", "kohakumoss_spec": json.dumps(asdict(self.spec))}
        files.write(path, files.to_storage(self.state_dict(), None), meta)

    def merge_into(self, multiplier: float = 1.0) -> None:
        """Bake every ΔW into its base weight and detach the adapters."""
        stored = [t.path for t in self.targets if t.stored]
        if stored:
            raise RuntimeError(
                f"cannot merge into quantized/streamed base storage: {stored[:4]}; "
                "export a merged checkpoint instead"
            )
        with torch.no_grad():
            for t in self.targets:
                dw, db = delta(t, multiplier)
                w = t.base_weight(torch.float32)
                bias = t.base_bias()
                self._detach(t)
                t.linear.weight.data.copy_(
                    (w + dw.to(w.device)).to(t.linear.weight.dtype)
                )
                if db is not None:
                    base_b = 0.0 if bias is None else bias.float().to(db.device)
                    add_bias(t.linear, base_b + db)
            for n in self.norms:
                norm_delta.detach(n.module, True, "weight")
                if n.bias_delta is not None:
                    norm_delta.detach(n.module, True, "bias")
        self.targets, self.by_name, self.norms = [], {}, []

    def remove(self) -> None:
        """Detach every adapter; the base layers forward as before apply()."""
        for t in self.targets:
            self._detach(t)
        for n in self.norms:
            norm_delta.detach(n.module, False, "weight")
            if n.bias_delta is not None:
                norm_delta.detach(n.module, False, "bias")
        self.targets, self.by_name, self.norms = [], {}, []

    @staticmethod
    def _detach(t: Target) -> None:
        if t.kind == "full":
            t.lyco.restore()
            weight = t.lyco.org_weight.to(t.lyco.weight.device)
            t.linear.weight = nn.Parameter(weight, requires_grad=False)
            if t.lyco.org_bias:
                bias = t.lyco.org_bias[0].to(t.lyco.weight.device)
                t.linear.bias = nn.Parameter(bias, requires_grad=False)
            getattr(t.linear, "rebind", _noop)()
        vars(t.linear).pop("forward", None)
        vars(t.lyco).pop("train", None)
        if CHILD in t.linear._modules:
            del t.linear._modules[CHILD]
