"""Inference-time adapters: load files of any supported format, bypass or merge, undo exactly."""

import json
from functools import partial

import torch
import torch.nn.functional as F
from lycoris.modules import IA3Module, get_module, make_module
from torch import Tensor, nn

from kohakumoss.adapters.adapter_set import RAW_TAG
from kohakumoss.adapters.apply import apply, compute_device
from kohakumoss.adapters.delta import delta, resolve, unit_delta
from kohakumoss.adapters.export.convert import row_norm
from kohakumoss.adapters.formats import group, read, spec_from_meta, stem_index
from kohakumoss.adapters.network import _DenseWeight, lora_name_of
from kohakumoss.adapters.spec import AdapterSpec
from kohakumoss.adapters.target import Target, kind_of

MODES = ("bypass", "merge")
DENSE_IN_BYPASS = ("full", "dylora")


def _dense_add(prev, dw: Tensor, x):
    return prev(x) + F.linear(x, dw.to(x.dtype))


def _dense_add_bias(prev, dw: Tensor, db: Tensor, x):
    return prev(x) + F.linear(x, dw.to(x.dtype), db.to(x.dtype))


def _lyco_bypass(lyco, multiplier, x):
    return lyco.bypass_forward(x, multiplier).to(x.dtype)


def _default_alpha(parts: dict) -> Tensor:
    """A missing alpha means scale 1 (ComfyUI); OFT/BOFT then has no constraint."""
    for name, axis in (
        ("lora_down.weight", 0),
        ("hada_w1_b", 0),
        ("lokr_w1_b", 0),
        ("lokr_w2_b", 0),
        ("a2.weight", 0),
    ):
        if name in parts:
            return torch.tensor(float(parts[name].shape[axis]))
    return torch.tensor(0.0)


def _rows(s: Tensor, like: Tensor) -> Tensor:
    return s.to(device=like.device, dtype=torch.float32).reshape(
        -1, *[1] * (like.dim() - 1)
    )


def _scaled_bypass(prev, lyco, multiplier, scale, x):
    diff = lyco.bypass_forward_diff(x, multiplier)
    return prev(x) + diff * scale.to(diff.dtype)


def _block_uniform(scale: Tensor, block: int) -> bool:
    rows = scale.reshape(-1, block)
    return bool((rows == rows[:, :1]).all())


class _Unfolded:
    """Context dividing a folded linear's weight rows by its fold scale (restored exactly)."""

    def __init__(self, linear: nn.Module, scale: Tensor | None):
        self.linear, self.scale, self.saved = linear, scale, None

    def __enter__(self):
        if self.scale is not None:
            w = self.linear.weight
            self.saved = w.detach().clone()
            w.data.copy_((w.float() / _rows(self.scale, w)).to(w.dtype))
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            self.linear.weight.data.copy_(self.saved)
        return False


ADDITIVE = ("locon", "loha", "lokr", "tlora")
COMMUTING = ("oft", "boft", "ia3")


class AdapterStack:
    """Adapters loaded for inference; `remove()` restores weights and forwards bit-exactly.

    `row_scales` is the residual fold's per-output-row scale of a folded model
    (model.numerics.output_row_scales); files hold unfolded ΔW, so their rows are
    multiplied by it.
    """

    def __init__(
        self, model: nn.Module, mode: str = "bypass", row_scales: dict | None = None
    ):
        if mode not in MODES:
            raise KeyError(f"mode {mode!r} not in {MODES}")
        self.model = model
        self.mode = mode
        self.device = compute_device(model)
        self.row_scales = dict(row_scales or {})
        self.modules = dict(model.named_modules())
        self.index = stem_index(model)
        self.originals: dict[tuple[int, str], tuple[nn.Module, str, Tensor]] = {}
        self.forwards: dict[int, tuple[nn.Module, object]] = {}
        self.live: list[nn.Module] = []
        self.sets = []

    def _save_original(self, module: nn.Module, name: str) -> None:
        key = (id(module), name)
        if key not in self.originals:
            param = getattr(module, name)
            self.originals[key] = (module, name, param.detach().cpu().clone())

    def _add_to(self, module: nn.Module, name: str, value: Tensor) -> None:
        self._save_original(module, name)
        param = getattr(module, name)
        param.data.add_(
            value.to(device=param.device, dtype=torch.float32).to(param.dtype)
        )

    def _chain(self, linear: nn.Module, make) -> None:
        if id(linear) not in self.forwards:
            self.forwards[id(linear)] = (linear, vars(linear).get("forward"))
        prev = vars(linear).get("forward") or type(linear).forward.__get__(linear)
        vars(linear)["forward"] = make(prev)

    @torch.no_grad()
    def add(self, source, multiplier: float = 1.0) -> None:
        """Load one adapter file (or tensor dict) at `multiplier`."""
        tensors, meta = (source, {}) if isinstance(source, dict) else read(source)
        if meta.get(RAW_TAG):
            self._add_raw(tensors, meta, multiplier)
            return
        groups, unmatched = group(tensors, self.index)
        if unmatched:
            raise KeyError(
                f"{len(unmatched)} keys match no layer, e.g. {unmatched[:4]}"
            )
        ours_comfy = meta.get("kohakumoss_format") == "comfy"
        for path, parts in groups.items():
            self._add_layer(path, parts, multiplier, ours_comfy)

    def _add_layer(self, path: str, parts: dict, m: float, ours_comfy: bool) -> None:
        module = self.modules[path]
        if "w_norm" in parts:
            self._add_to(module, "weight", parts["w_norm"].float() * m)
            if "b_norm" in parts and getattr(module, "bias", None) is not None:
                self._add_to(module, "bias", parts["b_norm"].float() * m)
            return
        scale = self.row_scales.get(path)
        if "diff" in parts:
            dw = parts["diff"].float() * m
            db = parts.get("diff_b")
            db = None if db is None else db.float() * m
            self._add_dense(module, *self._fold(dw, db, scale), 1.0)
            return
        stored = getattr(module, "base", None) is not None
        override = None
        if scale is not None:
            w = module.base_weight(torch.float32)
            override = w / _rows(scale, w)
        with _Unfolded(module, None if stored else scale):
            lyco = self._build(module, path, parts, ours_comfy, override)
        target = Target(
            path=path,
            lora_name=lyco.lora_name,
            linear=module,
            lyco=lyco,
            kind=kind_of(lyco),
            wd=bool(getattr(lyco, "wd", False)),
            wd_on_out=bool(getattr(lyco, "wd_on_out", True)),
            base_override=override,
        )
        resolve(target)
        dense = target.kind in DENSE_IN_BYPASS or (target.kind == "oft" and m != 1.0)
        if scale is not None and target.kind not in ADDITIVE:
            block = getattr(lyco, "block_size", 1)
            dense = dense or target.kind not in COMMUTING
            dense = dense or not _block_uniform(scale, block)
        if self.mode == "merge" or target.wd or dense:
            dw, db = delta(target, m)
            self._add_dense(module, *self._fold(dw, db, scale), 1.0)
            return
        lyco.multiplier = m
        lyco.bypass_mode = True
        self.live.append(lyco)
        if scale is not None and target.kind in ADDITIVE:
            s = scale.to(self.device).float()
            self._chain(module, lambda prev: partial(_scaled_bypass, prev, lyco, m, s))
            return

        def make(prev):
            lyco.org_forward = prev
            return partial(_lyco_bypass, lyco, m)

        self._chain(module, make)

    @staticmethod
    def _fold(dw: Tensor, db, scale):
        if scale is None:
            return dw, db
        dw = dw * _rows(scale, dw)
        return dw, None if db is None else db * _rows(scale, db).flatten()

    def _add_dense(self, module: nn.Module, dw: Tensor, db, m: float) -> None:
        """Merge ΔW into a dense weight, or chain x ΔW^T onto the forward of stored ones."""
        db = None if db is None else db.float() * m
        if self.mode == "merge" and getattr(module, "base", None) is None:
            self._add_to(module, "weight", dw)
            if db is not None:
                self._add_to(module, "bias", db)
            return
        dw, db = dw.to(self.device), None if db is None else db.to(self.device)
        if db is None:
            self._chain(module, lambda prev: partial(_dense_add, prev, dw))
        else:
            self._chain(module, lambda prev: partial(_dense_add_bias, prev, dw, db))

    def _build(
        self,
        linear: nn.Module,
        path: str,
        parts: dict,
        ours_comfy: bool,
        override: Tensor | None = None,
    ):
        name = lora_name_of("lora_unet", path)
        parts = dict(parts)
        parts.setdefault("alpha", _default_alpha(parts))
        dora_scale = parts.pop("dora_scale", None)
        sd = {f"{name}.{k}": v for k, v in parts.items()}
        lyco_type, params = get_module(sd, name)
        if lyco_type is None:
            raise KeyError(f"{path}: no LyCORIS algorithm matches {sorted(parts)}")
        dense = getattr(linear, "base", None) is not None
        ctx = _DenseWeight(linear) if dense else None
        if ctx is not None:
            ctx.__enter__()
        try:
            if lyco_type is IA3Module:
                lyco = IA3Module(
                    name, linear, 1.0, train_on_input=bool(parts["on_input"])
                )
                lyco.weight.data.copy_(parts["weight"])
            else:
                lyco = make_module(lyco_type, params, name, linear)
            if lyco is None:
                raise KeyError(f"{path}: {lyco_type.__name__} cannot load from a file")
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        if dora_scale is not None:
            lyco.wd = True
            lyco.wd_on_out = dora_scale.shape[0] == lyco.shape[0]
            lyco.dora_scale = nn.Parameter(dora_scale.float())
        lyco.to(device=self.device, dtype=torch.float32)
        lyco.requires_grad_(False)
        lyco.eval()
        if ours_comfy:
            self._undo_comfy(linear, lyco, parts, override)
        return lyco

    def _undo_comfy(self, linear, lyco, parts, override) -> None:
        """Invert the ComfyUI-semantics conversions of our own comfy exports."""
        kind = kind_of(lyco)
        if kind in ("oft", "boft") and "alpha" in parts:
            lyco.constraint = float(parts["alpha"])
        if getattr(lyco, "wd", False) and lyco.wd_on_out:
            t = Target("", lyco.lora_name, linear, lyco, kind, True, True)
            t.set_override(override)
            resolve(t)
            w = t.base_weight(torch.float32)
            dw = unit_delta(t).to(w.device)
            eps = torch.finfo(torch.float32).eps
            ratio = (row_norm(w + dw) + eps) / (row_norm(w) + eps)
            lyco.dora_scale.data.mul_(ratio.to(lyco.dora_scale))

    def _add_raw(self, tensors: dict, meta: dict, m: float) -> None:
        """A raw training state: rebuild the adapter set from its spec, then bind or merge."""
        fields = spec_from_meta(meta)
        quiet = {"dropout": 0.0, "rank_dropout": 0.0, "module_dropout": 0.0}
        spec = AdapterSpec(**{**fields, **quiet, "multiplier": m})
        aset = apply(self.model, spec)
        aset.load_state_dict(tensors)
        for p in aset.trainable_params():
            p.requires_grad_(False)
        for t in aset.targets:
            t.lyco.eval()
        if self.mode == "merge":
            deltas = [(t.linear, *delta(t, m)) for t in aset.targets]
            norm_deltas = [(n.path, n.delta()) for n in aset.norms]
            aset.remove()
            for linear, dw, db in deltas:
                self._add_dense(linear, dw, db, 1.0)
            for path, (dw, db) in norm_deltas:
                parts = {"w_norm": dw} if db is None else {"w_norm": dw, "b_norm": db}
                self._add_layer(path, parts, m, False)
            return
        for t in aset.targets:
            self.forwards.setdefault(id(t.linear), (t.linear, None))
        self.sets.append(aset)

    def remove(self) -> None:
        """Restore every touched weight and forward exactly."""
        with torch.no_grad():
            for aset in self.sets:
                aset.remove()
            for module, previous in self.forwards.values():
                vars(module).pop("forward", None)
                if previous is not None:
                    vars(module)["forward"] = previous
            for module, name, value in self.originals.values():
                getattr(module, name).data.copy_(value)
        self.originals, self.forwards, self.live, self.sets = {}, {}, [], []

    unmerge = remove


def load_adapters(
    model: nn.Module,
    adapters: list[tuple[str | dict, float]],
    mode: str = "bypass",
    row_scales: dict | None = None,
) -> AdapterStack:
    """Load each (file or tensor dict, multiplier) onto `model` for inference."""
    stack = AdapterStack(model, mode, row_scales)
    for source, multiplier in adapters:
        stack.add(source, multiplier)
    return stack


def load(
    model: nn.Module,
    source: str | dict,
    multiplier: float = 1.0,
    mode: str = "bypass",
    row_scales: dict | None = None,
) -> AdapterStack:
    """Load one adapter; stacks with adapters loaded before it on the same layers."""
    return load_adapters(model, [(source, multiplier)], mode, row_scales)


def describe(path: str) -> dict:
    """Format, key style and spec recorded in an adapter file's metadata."""
    _, meta = read(path)
    spec = meta.get("kohakumoss_spec")
    return {
        "format": meta.get("kohakumoss_format", "unknown"),
        "style": meta.get("kohakumoss_key_style", "unknown"),
        "spec": json.loads(spec) if spec else None,
    }
