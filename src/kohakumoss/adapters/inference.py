"""Inference-time adapters: load files of any supported format, bypass or merge, undo exactly."""

import json
from functools import partial

import lycoris.kernels as lyk
import torch
from lycoris.modules import IA3Module, get_module, make_module
from torch import Tensor, nn

from kohakumoss.adapters.adapter_set import RAW_TAG
from kohakumoss.adapters.apply import apply, compute_device
from kohakumoss.adapters.backend import pick_backend
from kohakumoss.adapters.bind.compose import dense_of
from kohakumoss.adapters.export.convert import row_norm
from kohakumoss.adapters.formats import group, read, spec_from_meta, stem_index
from kohakumoss.adapters.kinds import record_of
from kohakumoss.adapters.network import _DenseWeight, lora_name_of
from kohakumoss.adapters.scope import compute_dtype_of, rebind_owner
from kohakumoss.adapters.spec import inference_spec
from kohakumoss.adapters.target import Target

MODES = ("bypass", "merge")


def _dense_add(prev, dense, dw: Tensor, x):
    return prev(x) + dense(x, dw)


def _dense_add_bias(dense, dw: Tensor, db: Tensor, prev, x):
    y = dense(x, dw)
    return prev(x) + y + db.to(y.dtype)


def _noop() -> None:
    return None


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
    ):
        if mode not in MODES:
            raise KeyError(f"mode {mode!r} not in {MODES}")
        self.model = model
        self.mode = mode
        self.undoable = undoable
        self.device = compute_device(model)
        self.backend = pick_backend(self.device, "auto")
        lyk.set_half_matmul(compute_dtype_of(model.modules()))
        self.row_scales = dict(row_scales or {})
        self.modules = dict(model.named_modules())
        self.index = stem_index(model)
        self.originals: dict[tuple[int, str], tuple[nn.Module, str, Tensor]] = {}
        self.forwards: dict[int, tuple[nn.Module, object]] = {}
        self.earlier: dict[str, list] = {}
        self.live: list[nn.Module] = []
        self.created_bias: list[nn.Module] = []

    def _ensure_bias(self, module: nn.Module) -> None:
        """Give a bias-less linear a zero bias (removed again by remove())."""
        if module.bias is not None:
            return
        dtype = module.weight.dtype if module.weight.numel() else torch.float32
        zeros = torch.zeros(module.out_features, device=self.device, dtype=dtype)
        module.bias = nn.Parameter(zeros, requires_grad=False)
        getattr(module, "rebind", _noop)()
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
        else:
            groups, unmatched = group(tensors, self.index)
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
        if "diff" in parts:
            dw = parts["diff"].float() * m
            db = parts.get("diff_b")
            db = None if db is None else db.float() * m
            self._add_dense(path, module, dw, db, scale)
            return
        stored = getattr(module, "base", None) is not None
        with _Unfolded(module, None if stored else scale):
            lyco = self._build(module, path, parts)
        override = self._base_for(module, path, lyco, m)
        if ours_comfy:
            self._undo_comfy(module, lyco, parts, override)
        self._attach(path, module, lyco, m, override, scale)

    def _base_for(self, module, path: str, lyco, m: float) -> Tensor | None:
        """The effective-weight override, built only for a dW that reads its base."""
        rec = record_of(lyco)
        reads_base = getattr(lyco, "wd", False) or rec.stacking != "additive"
        if reads_base or not rec.exact_at(m):
            return self._unfolded_base(module, path)
        return None

    def _attach(self, path, module, lyco, m: float, override, scale) -> None:
        """Merge, chain densely, or chain the module's bypass for one built module."""
        lyco.kernel_backend = self.backend
        target = Target(path, lyco.lora_name, module, lyco, override, self.backend)
        rec = target.record
        dense = rec.stacking == "dense" or not rec.exact_at(m)
        if scale is not None and rec.stacking != "additive":
            block = getattr(lyco, "block_size", 1)
            uniform = rec.stacking == "commuting" and _block_uniform(scale, block)
            dense = dense or not uniform
        if self.mode == "merge" or target.wd or dense:
            dw, db = target.delta(m)
            self._add_dense(path, module, dw, db, scale)
            return
        lyco.multiplier = m
        lyco.bypass_mode = True
        self.live.append(lyco)
        self.earlier.setdefault(path, []).append(partial(_first, target.delta_at(m)))
        if scale is not None and rec.stacking == "additive":
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

    def _add_dense(self, path: str, module: nn.Module, dw: Tensor, db, scale) -> None:
        """Merge (unfolded dW, db) into a dense weight, or chain x dW^T onto the forward,
        held once in the compute dtype."""
        unfolded = dw
        dw, db = self._fold(dw, db, scale)
        if self.mode == "merge" and getattr(module, "base", None) is None:
            self._add_to(module, "weight", dw)
            if db is not None:
                self._ensure_bias(module)
                self._add_to(module, "bias", db)
            return
        self.earlier.setdefault(path, []).append(partial(_held, unfolded.float()))
        dtype = getattr(module, "input_dtype", None) or module.weight.dtype
        dw = dw.to(self.device, dtype)
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
        return lyco

    def _undo_comfy(self, linear, lyco, parts, override) -> None:
        """Invert the ComfyUI-semantics conversions of our own comfy exports."""
        t = Target("", lyco.lora_name, linear, lyco, override, self.backend)
        t.record.from_comfy(lyco, parts)
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

    def _add_raw(self, tensors: dict, meta: dict, m: float) -> None:
        """A raw training state: its modules rebuilt from its spec (dropout off, the
        device's own tier), then stacked like any loaded adapter."""
        spec = inference_spec(spec_from_meta(meta) or {}, 1.0)
        stash = self._stash_forwards()
        try:
            aset = apply(self.model, spec, device=self.device)
            aset.load_state_dict(tensors)
            built = [(t.path, t.linear, t.lyco) for t in aset.targets]
            norm_deltas = [(n.path, n.delta()) for n in aset.norms]
            aset.remove()
        finally:
            self._unstash(stash)
        for path, linear, lyco in built:
            lyco.requires_grad_(False)
            lyco.eval()
            override = self._base_for(linear, path, lyco, m)
            self._attach(path, linear, lyco, m, override, None)
        for path, (dw, db) in norm_deltas:
            parts = {"w_norm": dw} if db is None else {"w_norm": dw, "b_norm": db}
            self._add_norm(self.modules[path], parts, m)

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
                getattr(module, "rebind", _noop)()
        self.originals, self.forwards, self.live, self.earlier = {}, {}, [], {}
        self.created_bias = []
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
) -> AdapterStack:
    """Load each (file or tensor dict, multiplier) onto `model` for inference."""
    stack = AdapterStack(model, mode, row_scales, undoable)
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
