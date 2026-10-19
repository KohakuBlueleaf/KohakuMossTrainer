"""Inference-time adapters: load files of any supported format, bypass or merge, undo exactly."""

import json
import weakref
from functools import partial

import torch
from lycoris.modules import IA3Module, get_module, make_module
from torch import Tensor, nn

from kohakumoss.adapters.bind.compose import dense_of
from kohakumoss.adapters.build.apply import apply, compute_device
from kohakumoss.adapters.build.config import inference_config
from kohakumoss.adapters.build.network import lora_name_of
from kohakumoss.adapters.build.target import (
    Target,
    base_forward,
    base_kind,
    is_stored,
    weak_org,
)
from kohakumoss.adapters.export.convert import row_norm
from kohakumoss.adapters.export.fused import split_qkv
from kohakumoss.adapters.infer import stored
from kohakumoss.adapters.infer.formats import config_from_meta, group, read, stem_index
from kohakumoss.adapters.kinds import record_of
from kohakumoss.adapters.kinds.record import Stacking
from kohakumoss.adapters.lycoris.backend import pick_backend
from kohakumoss.adapters.train.adapter_set import (
    FULL_TAG,
    RAW_TAG,
    check_adapter_file,
    ensure_bias,
    fold_of,
)
from kohakumoss.adapters.train.scope import rebind_owner
from kohakumoss.logging.log import get_logger, hl
from kohakumoss.model.numerics import record_row_scales
from kohakumoss_common.calls import nothing

log = get_logger(__name__)

MODES = ("bypass", "merge")


def _dense_add(prev, dense, dw: Tensor, x):
    return prev(x) + dense(x, dw)


def _dense_add_bias(dense, dw: Tensor, db: Tensor, prev, x):
    y = dense(x, dw)
    return prev(x) + y + db.to(y.dtype)


def _lyco_bypass(prev, lyco, multiplier, x):
    return prev(x) + lyco.bypass_forward_diff(x, multiplier).to(x.dtype)


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


def _same_rows(a: Tensor | None, b: Tensor | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.shape == b.shape and bool(torch.equal(a.float().cpu(), b.float().cpu()))


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


class AdapterStack:
    """Adapters loaded for inference; `remove()` restores weights and forwards bit-exactly.

    `row_scales` is the residual fold's per-output-row scale of a folded model; files
    hold unfolded dW, so their rows are multiplied by it. `undoable=False` keeps no
    copy of merged weights (a one-way merge; `remove()` then raises).
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "bypass",
        row_scales: dict | None = None,
        undoable: bool = True,
        device: torch.device | str | None = None,
    ):
        if mode not in MODES:
            raise KeyError(f"mode {mode!r} not in {MODES}")
        self.model = model
        self.mode = mode
        self.undoable = undoable
        self.device = torch.device(device) if device else compute_device(model)
        self.backend = pick_backend(self.device, "auto")
        self.row_scales = dict(row_scales or {})
        self.modules = dict(model.named_modules())
        self.index = stem_index(model)
        self.originals: dict[tuple[int, str], tuple[nn.Module, str, Tensor]] = {}
        self.forwards: dict[int, tuple[nn.Module, object]] = {}
        self.earlier: dict[str, list] = {}
        self.live: list[nn.Module] = []
        self.created_bias: list[nn.Module] = []
        self.bypassed = 0

    @property
    def effective_mode(self) -> str:
        """The mode the loaded layers run: `mode`, or "merge + bypass" when a merge
        request met quantized / streamed bases (those layers run bypass)."""
        if self.mode == "merge" and self.bypassed:
            return f"merge + bypass ({self.bypassed} layers on stored bases)"
        return self.mode

    def _ensure_bias(self, module: nn.Module) -> None:
        """Give a bias-less linear a zero bias (removed again by remove())."""
        if ensure_bias(module, self.device):
            self.created_bias.append(module)

    def _save_original(self, module: nn.Module, name: str) -> None:
        key = (id(module), name)
        if self.undoable and key not in self.originals:
            param = getattr(module, name)
            self.originals[key] = (module, name, param.detach().cpu().clone())

    def _add_to(self, module: nn.Module, name: str, value: Tensor) -> None:
        self._save_original(module, name)
        param = getattr(module, name)
        merged = param.float() + value.to(device=param.device, dtype=torch.float32)
        param.data.copy_(merged.to(param.dtype))

    def _chain(self, linear: nn.Module, make) -> None:
        """`make(prev)` as the layer's forward: `prev` its current one, or its class
        forward over a weak reference."""
        if id(linear) not in self.forwards:
            self.forwards[id(linear)] = (linear, vars(linear).get("forward"))
        prev = vars(linear).get("forward") or partial(base_forward, weakref.ref(linear))
        vars(linear)["forward"] = make(prev)

    @torch.no_grad()
    def add(self, source, multiplier: float = 1.0) -> None:
        """Load one adapter file (or tensor dict) at `multiplier`; a training state (its
        tag, spec and fold in the metadata; a tagged file missing one, or an untagged
        file inside a run checkpoint, is refused) is converted to its unfolded LyCORIS
        tensors first, a fused q/k/v layer is split.
        """
        if isinstance(source, dict):
            tensors, meta = source, {}
        else:
            tensors, meta = read(source)
            check_adapter_file(source, meta)
        if meta.get(FULL_TAG):
            raise ValueError(
                f"{source}: a full-finetune checkpoint holds the backbone's weights, not "
                "an adapter; export it as a merged transformer (scripts/export/export.py"
                " FORMAT='merged') and load that as the DiT"
            )
        if meta.get(RAW_TAG):
            tensors = self._raw_tensors(tensors, meta)
        groups, unmatched = group(split_qkv(tensors), self.index)
        if unmatched:
            raise KeyError(
                f"{len(unmatched)} keys match no layer, e.g. {unmatched[:4]}"
            )
        ours_comfy = meta.get("kohakumoss_format") == "comfy"
        for path, parts in groups.items():
            self._add_layer(path, parts, multiplier, ours_comfy)
        rebind_owner(self.model)

    def _add_norm(self, module: nn.Module, parts: dict, m: float) -> None:
        self._add_to(module, "weight", parts["w_norm"].float() * m)
        if "b_norm" in parts and getattr(module, "bias", None) is not None:
            self._add_to(module, "bias", parts["b_norm"].float() * m)

    def _unfolded_base(self, module: nn.Module, path: str) -> Tensor | None:
        """The layer's current effective weight in unfolded space, when it differs
        from what its own storage reads (a fold scale, or earlier bypass adapters)."""
        scale = self.row_scales.get(path)
        earlier = self.earlier.get(path, [])
        if scale is None and not earlier:
            return None
        w = module.base_weight(torch.float32)
        if scale is not None:
            w = w / _rows(scale, w)
        for dw in earlier:
            w = w + dw().to(w.device)
        return w

    def _add_layer(self, path: str, parts: dict, m: float, ours_comfy: bool) -> None:
        module = self.modules[path]
        if "w_norm" in parts:
            self._add_norm(module, parts, m)
            return
        scale = self.row_scales.get(path)
        on_stored = is_stored(module)
        if "diff" in parts:
            if on_stored:
                raise ValueError(
                    f"{path}: a dense-diff adapter needs a 16/32-bit dense base; this "
                    f"layer is {base_kind(module)} (load it on a compute-precision model)"
                )
            dw = parts["diff"].float() * m
            db = parts.get("diff_b")
            db = None if db is None else db.float() * m
            self._add_dense(path, module, dw, db, scale)
            return
        with _Unfolded(module, None if on_stored else scale):
            lyco = self._build(module, path, parts)
        if on_stored:
            if ours_comfy:
                self._undo_comfy(module, lyco, parts, self._unfolded_base(module, path))
            self._attach_stored(path, module, lyco, m, scale)
            return
        override = self._base_for(module, path, lyco, m)
        if ours_comfy:
            self._undo_comfy(module, lyco, parts, override)
        self._attach(path, module, lyco, m, override, scale)

    def _base_for(self, module, path: str, lyco, m: float) -> Tensor | None:
        """The effective-weight override, built only for a dW that reads its base."""
        rec = record_of(lyco)
        reads_base = getattr(lyco, "wd", False) or rec.stacking != Stacking.ADDITIVE
        if reads_base or not rec.bypass_exact_at(m):
            return self._unfolded_base(module, path)
        return None

    def _attach(self, path, module, lyco, m: float, override, scale) -> None:
        """Merge, chain densely, or chain the module's bypass for one built module."""
        target = Target(path, lyco.lora_name, module, lyco, override, self.backend)
        rec = target.record
        dense = rec.stacking == Stacking.DENSE or not rec.bypass_exact_at(m)
        if scale is not None and rec.stacking != Stacking.ADDITIVE:
            block = getattr(lyco, "block_size", 1)
            uniform = rec.stacking == Stacking.COMMUTING and _block_uniform(
                scale, block
            )
            dense = dense or not uniform
        if self.mode == "merge" or target.wd or dense:
            dw, db = target.delta(m)
            self._add_dense(path, module, dw, db, scale)
            return
        lyco.multiplier = m
        lyco.bypass_mode = True
        self.live.append(lyco)
        self.earlier.setdefault(path, []).append(partial(_first, target.delta_at(m)))
        if scale is not None and rec.stacking == Stacking.ADDITIVE:
            s = scale.to(self.device).float()
            self._chain(module, lambda prev: partial(_scaled_bypass, prev, lyco, m, s))
            return

        def make(prev):
            lyco.org_forward = prev
            return partial(_lyco_bypass, prev, lyco, m)

        self._chain(module, make)

    def _attach_stored(self, path, module, lyco, m: float, scale) -> None:
        """Chain the module's bypass onto a quantized / streamed base, whatever the
        mode: its delta in unfolded space, times the fold's row scale."""
        if self.mode == "merge" and not self.bypassed:
            log.warning(
                f"ADAPTER_MODE='merge': adapters on a {hl(base_kind(module))} base run "
                "as bypass (a merge needs a 16/32-bit dense base)"
            )
        self.bypassed += 1
        lyco.multiplier = m
        lyco.bypass_mode = True
        self.live.append(lyco)
        target = Target(path, lyco.lora_name, module, lyco, None, self.backend)
        s = None if scale is None else scale.to(self.device).float().reshape(-1)
        if target.wd:
            self._chain(module, self._stored_dora(path, module, target, m, s))
        else:
            bias_u = stored.diff_bias(lyco, module, s)
            self._chain(module, lambda prev: stored.wire(prev, lyco, m, s, bias_u))
        self.earlier.setdefault(path, []).append(partial(_first, target.delta_at(m)))

    def _stored_dora(self, path, module, target, m: float, s):
        """The DoRA chain of a stored base, its ratio taken once from one transient
        copy of the effective unfolded weight."""
        w = self._unfolded_base(module, path)
        if w is None:
            w = module.base_weight(torch.float32)
        ratio = stored.dora_ratio(target.lyco, w.float(), target.unit_fn())
        del w
        lyco = target.lyco
        bias = None
        if lyco.wd_on_out and module.bias is not None:
            bias = module.bias.detach().float()
        return lambda prev: stored.wire_dora(prev, lyco, m, ratio, s, bias)

    @staticmethod
    def _fold(dw: Tensor, db, scale):
        if scale is None:
            return dw, db
        dw = dw * _rows(scale, dw)
        return dw, None if db is None else db * _rows(scale, db).flatten()

    def _add_dense(self, path: str, module: nn.Module, dw: Tensor, db, scale) -> None:
        """Merge (unfolded dW, db) into a dense weight, or chain x dW^T onto the forward,
        held once in the layer's weight dtype."""
        unfolded = dw
        dw, db = self._fold(dw, db, scale)
        if self.mode == "merge" and getattr(module, "base", None) is None:
            self._add_to(module, "weight", dw)
            if db is not None:
                self._ensure_bias(module)
                self._add_to(module, "bias", db)
            return
        self.earlier.setdefault(path, []).append(partial(_held, unfolded.float()))
        dw = dw.to(self.device, module.weight.dtype)
        dense = dense_of(module)
        if db is None:
            self._chain(module, lambda prev: partial(_dense_add, prev, dense, dw))
            return
        db = db.to(self.device, torch.float32)
        add = partial(_dense_add_bias, dense, dw, db)
        self._chain(module, lambda prev: partial(add, prev))

    def _build(self, linear: nn.Module, path: str, parts: dict):
        name = lora_name_of("lora_unet", path)
        parts = dict(parts)
        parts.setdefault("alpha", _default_alpha(parts))
        dora_scale = parts.pop("dora_scale", None)
        sd = {f"{name}.{k}": v for k, v in parts.items()}
        lyco_type, params = get_module(sd, name)
        if lyco_type is None:
            raise KeyError(f"{path}: no LyCORIS algorithm matches {sorted(parts)}")
        if lyco_type is IA3Module:
            lyco = IA3Module(name, linear, 1.0, train_on_input=bool(parts["on_input"]))
            lyco.weight.data.copy_(parts["weight"])
        elif is_stored(linear):
            lyco = stored.build(lyco_type, params, name, linear)
        else:
            lyco = make_module(lyco_type, params, name, linear)
        if lyco is None:
            raise KeyError(f"{path}: {lyco_type.__name__} cannot load from a file")
        weak_org(lyco, linear)
        if dora_scale is not None:
            lyco.wd = True
            lyco.wd_on_out = dora_scale.shape[0] == lyco.shape[0]
            lyco.dora_scale = nn.Parameter(dora_scale.float())
        lyco.to(device=self.device, dtype=torch.float32)
        lyco.requires_grad_(False)
        lyco.eval()
        return lyco

    def _undo_comfy(self, linear, lyco, parts, override) -> None:
        """Invert the ComfyUI-semantics conversions of our own comfy exports."""
        t = Target("", lyco.lora_name, linear, lyco, override, self.backend)
        t.record.undo_comfy(lyco, parts)
        if t.wd and t.wd_on_out:
            w = t.base_weight(torch.float32)
            dw = t.unit_fn().to(w.device)
            eps = torch.finfo(torch.float32).eps
            ratio = (row_norm(w + dw) + eps) / (row_norm(w) + eps)
            lyco.dora_scale.data.mul_(ratio.to(lyco.dora_scale))

    def _stash_forwards(self) -> dict[int, tuple[nn.Module, object]]:
        out = {}
        for m in self.model.modules():
            fwd = vars(m).pop("forward", None)
            if fwd is not None:
                out[id(m)] = (m, fwd)
        return out

    @staticmethod
    def _unstash(stash: dict) -> None:
        for m, fwd in stash.values():
            vars(m)["forward"] = fwd

    def _source_base(self, t: Target, source: dict) -> Tensor | None:
        """The effective weight of `t`'s layer in the space of a state trained under
        the fold whose row scales are `source` (unfolded, times that scale); None when
        the layer's own storage reads exactly that."""
        src = source.get(t.path)
        if _same_rows(self.row_scales.get(t.path), src) and not self.earlier.get(
            t.path
        ):
            return None
        w = self._unfolded_base(t.linear, t.path)
        if w is None:
            w = t.linear.base_weight(torch.float32)
        return w if src is None else w * _rows(src, w)

    def _raw_tensors(self, tensors: dict, meta: dict) -> dict[str, Tensor]:
        """A raw training state as unfolded LyCORIS tensors keyed by layer path: its
        modules rebuilt from its configuration (dropout off, the device's own backend),
        each layer's dW rows divided by the fold the state trained on (the recorded
        fold; this model's own when the state records none)."""
        config = inference_config(config_from_meta(meta) or {}, 1.0)
        recorded, record = fold_of(meta)
        source = record_row_scales(self.model, record) if recorded else self.row_scales
        stash = self._stash_forwards()
        try:
            aset = apply(self.model, config, device=self.device)
            aset.load_state_dict(tensors)
            base_of = partial(self._source_base, source=source)
            raw = aset.tensors("lycoris", row_scales=source, base_of=base_of)
            paths = {t.lora_name: t.path for t in aset.targets}
            paths.update({n.lora_name: n.path for n in aset.norms})
            aset.remove()
        finally:
            self._unstash(stash)
        out = {}
        for key, value in raw.items():
            stem, _, name = key.partition(".")
            out[f"{paths[stem]}.{name}"] = value
        return out

    def remove(self) -> None:
        """Restore every touched weight and forward exactly."""
        if not self.undoable:
            raise RuntimeError("this AdapterStack was loaded with undoable=False")
        with torch.no_grad():
            for module, previous in self.forwards.values():
                vars(module).pop("forward", None)
                if previous is not None:
                    vars(module)["forward"] = previous
            for module, name, value in self.originals.values():
                getattr(module, name).data.copy_(value)
            for module in self.created_bias:
                module.bias = None
                getattr(module, "rebind", nothing)()
        self.originals, self.forwards, self.live, self.earlier = {}, {}, [], {}
        self.created_bias, self.bypassed = [], 0
        rebind_owner(self.model)

    unmerge = remove


def _first(builder):
    return builder()[0]


def _held(tensor: Tensor) -> Tensor:
    return tensor


def load_adapters(
    model: nn.Module,
    adapters: list[tuple[str | dict, float]],
    mode: str = "bypass",
    row_scales: dict | None = None,
    undoable: bool = True,
    device: torch.device | str | None = None,
) -> AdapterStack:
    """Load each (file or tensor dict, multiplier) onto `model` for inference; the
    adapter modules live on `device` (default: the model's first non-empty parameter).
    """
    stack = AdapterStack(model, mode, row_scales, undoable, device)
    for source, multiplier in adapters:
        stack.add(source, multiplier)
    return stack


def load(
    model: nn.Module,
    source: str | dict,
    multiplier: float = 1.0,
    mode: str = "bypass",
    row_scales: dict | None = None,
    device: torch.device | str | None = None,
) -> AdapterStack:
    """Load one adapter; stacks with adapters loaded before it on the same layers."""
    return load_adapters(model, [(source, multiplier)], mode, row_scales, True, device)


def describe(path: str) -> dict:
    """Format, key style and adapter config recorded in an adapter file's metadata."""
    _, meta = read(path)
    config = meta.get("kohakumoss_spec")
    return {
        "format": meta.get("kohakumoss_format", "unknown"),
        "style": meta.get("kohakumoss_key_style", "unknown"),
        "config": json.loads(config) if config else None,
    }
