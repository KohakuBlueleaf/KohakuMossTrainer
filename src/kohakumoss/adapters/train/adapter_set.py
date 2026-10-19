"""LycoAdapterSet: the AdapterSet over LyCORIS modules bound to a model."""

import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn

from kohakumoss.adapters.bind.select import bind_target, training_dependent
from kohakumoss.adapters.build.config import AdapterConfig
from kohakumoss.adapters.build.target import CHILD, NormTarget, Target
from kohakumoss.adapters.export import convert, files
from kohakumoss.adapters.export.fused import (
    fused_diff,
    fused_lora,
    is_lora,
    qkv_groups,
)
from kohakumoss.adapters.export.merged import merged_state_dict
from kohakumoss.adapters.train import groups
from kohakumoss.adapters.train import norms as norm_delta
from kohakumoss.adapters.train.scope import evaluated, rebind_owner
from kohakumoss.adapters.train.steps import StepPlan
from kohakumoss.model.io.save import write_diffusers
from kohakumoss.model.numerics import fold_record
from kohakumoss_common.calls import nothing

FORMATS = ("lycoris", "comfy", "merged", "diffusers")
RAW_TAG = "kohakumoss_adapter_state"
FULL_TAG = "kohakumoss_full_state"
FOLD_KEY = "kohakumoss_fold"
FUSED_KEY = "kohakumoss_fused_qkv"
SPEC_KEY = "kohakumoss_spec"
STATE_FIELDS = {RAW_TAG: (SPEC_KEY, FOLD_KEY), FULL_TAG: (FOLD_KEY,)}
REFUSAL = "{path}: {what}; re-export from a run of this version"


def state_kind(path, meta: dict[str, str]) -> str:
    """RAW_TAG or FULL_TAG of a training-state file whose metadata carries its tag and
    STATE_FIELDS; ValueError (REFUSAL) naming what is missing otherwise."""
    kinds = [tag for tag in STATE_FIELDS if meta.get(tag)]
    if not kinds:
        what = f"no training-state metadata ({RAW_TAG} or {FULL_TAG})"
        raise ValueError(REFUSAL.format(path=path, what=what))
    missing = [name for name in STATE_FIELDS[kinds[0]] if name not in meta]
    if missing:
        what = f"training-state metadata without {', '.join(missing)}"
        raise ValueError(REFUSAL.format(path=path, what=what))
    return kinds[0]


def in_run(path) -> bool:
    """`path` is a file of a KohakuMoss run checkpoint: `<run>/checkpoints/<step>/<file>`
    with the run's `<run>/config.json`."""
    path = Path(path)
    run = path.parent.parent
    return run.name == "checkpoints" and (run.parent / "config.json").is_file()


def check_adapter_file(path, meta: dict[str, str]) -> None:
    """A file to load as an adapter: a training state (tagged) must be complete
    (`state_kind`), and an untagged file inside a run checkpoint is refused (REFUSAL);
    any other untagged file is a plain adapter file."""
    if any(meta.get(tag) for tag in STATE_FIELDS) or in_run(path):
        state_kind(path, meta)


def fold_metadata(model: nn.Module) -> dict[str, str]:
    """{FOLD_KEY: JSON of the weight fold `model` carries (null when unfolded)}: the
    space a training state's tensors live in."""
    return {FOLD_KEY: json.dumps(fold_record(model))}


def fold_of(meta: dict[str, str]) -> tuple[bool, dict | None]:
    """(recorded, fold record) of a training state's metadata; (False, None) for a
    file that records no fold."""
    if FOLD_KEY not in meta:
        return False, None
    return True, json.loads(meta[FOLD_KEY])


@contextmanager
def _based(t: Target, base_of: Callable | None):
    """`t` reads the base `base_of(t)` inside the block (its own when None)."""
    override = None if base_of is None else base_of(t)
    if override is None:
        yield
        return
    t.base_override = override
    t.resolve_base()
    try:
        yield
    finally:
        t.base_override = None
        t.resolve_base()


def ensure_bias(linear: nn.Module, device) -> bool:
    """Give a bias-less linear a frozen zero bias on `device` (re-selecting MossLinear's
    bias / no-bias forward); whether one was created."""
    if linear.bias is not None:
        return False
    dtype = linear.weight.dtype if linear.weight.numel() else torch.float32
    zeros = torch.zeros(linear.out_features, device=device, dtype=dtype)
    linear.bias = nn.Parameter(zeros, requires_grad=False)
    getattr(linear, "rebind", nothing)()
    return True


def add_bias(linear: nn.Module, value: Tensor) -> None:
    """Set the linear's bias to `value`, creating it on a bias-less layer."""
    ensure_bias(linear, value.device)
    linear.bias.data.copy_(value.to(linear.bias))


def _bind_one(t: Target, config: AdapterConfig, backend: str, multiplier: float):
    """Install `t`'s bound forward for these settings and its module's mode."""
    t.linear.forward = bind_target(
        t,
        config.mode,
        backend,
        multiplier,
        config.dora_detach,
        config.module_dropout,
        autotune=config.backend == "auto",
        fuse_base_add=config.fuse_base_add,
        dynamic=config.compile_dynamic,
    )


def _train_and_rebind(settings: tuple, t: Target, mode: bool = True):
    """Module.train for an adapted layer whose bound forward depends on the mode: the
    layer and its LyCORIS module change mode, then the forward is re-bound under
    `settings` (`_bind_one`'s config, backend, multiplier)."""
    linear = t.linear
    nn.Module.train(linear, mode)
    _bind_one(t, *settings)
    return linear


class LycoAdapterSet:
    """Adapters on `model`; forwards bound once, rebound on multiplier changes."""

    def __init__(
        self,
        model: nn.Module,
        config: AdapterConfig,
        targets: list[Target],
        norms: list[NormTarget],
        backend: str,
    ):
        self.model = model
        self.config = config
        self.targets = targets
        self.norms = norms
        self.backend = backend
        self.multiplier = config.multiplier
        self.steps = StepPlan(targets, config)
        self.prewarm_stats: dict[str, int] = {}
        self.bind()

    def bind(self) -> None:
        """Install every layer's forward for the current multiplier and training state;
        layers whose forward depends on train()/eval() re-bind through a hook on the
        layer (holding the settings and the target, not this set)."""
        config = self.config
        settings = (config, self.backend, self.multiplier)
        for t in self.targets:
            _bind_one(t, *settings)
            if training_dependent(t, config.module_dropout, config.dora_detach):
                vars(t.linear)["train"] = partial(_train_and_rebind, settings, t)
        for n in self.norms:
            for _, d in n.deltas():
                d.multiplier = self.multiplier
        rebind_owner(self.model)

    def bind_one(self, t: Target) -> None:
        _bind_one(t, self.config, self.backend, self.multiplier)

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)
        self.bind()

    def begin_step(self, sigma: Tensor, seq, step: int) -> None:
        """Before a training / validation forward: T-LoRA masks and this microbatch's
        DyLoRA / module-dropout draws."""
        self.steps.begin(sigma, seq, step)

    def end_step(self) -> None:
        """After the step's last backward: masks cleared (full-rank adapters)."""
        self.steps.end()

    def step_state(self):
        """The state the last `begin_step` installed (`set_step_state` puts it back)."""
        return self.steps.capture()

    def set_step_state(self, state) -> None:
        """Install a `step_state()` again (a micro-batch's later forward / recompute)."""
        self.steps.install(state)

    def prewarm(
        self,
        rows: list[int],
        dtype: torch.dtype,
        devices: list[torch.device],
        budgets: dict | None = None,
    ) -> int:
        """Settle every distinct autotuned op over `rows` on scratch inputs (no
        gradient left on the factors): ops with one `signature` are one op, and a
        problem already settled or stored is only bound; `budgets[device]` bounds a
        problem's timing there (`AutotunedOp.warm`). Returns the problems measured;
        `prewarm_stats` counts ops, distinct ops and problems measured."""
        budgets = budgets or {}
        allowed = set(devices)
        measured, ops_seen, seen = 0, 0, set()
        for t in self.targets:
            params = list(t.lyco.parameters())
            if not params or not t.autotuned or params[0].device not in allowed:
                continue
            device = params[0].device
            for op in t.autotuned:
                ops_seen += 1
                signature = op.signature(dtype, device)
                if signature in seen:
                    continue
                seen.add(signature)
                measured += op.warm(rows, dtype, device, budgets.get(device))
            for p in params:
                p.grad = None
        self.prewarm_stats = {
            "ops": ops_seen,
            "distinct": len(seen),
            "measured": measured,
        }
        return measured

    @staticmethod
    def _norm_params(n: NormTarget):
        return [(name, d.w_norm) for name, d in n.deltas()]

    def trainable_params(self):
        for _, p in self.named_trainable():
            yield p

    def named_trainable(self):
        for full, _, p, _ in self._entries():
            yield full, p

    def _entries(self):
        for t in self.targets:
            for name, p in t.lyco.named_parameters():
                if p.requires_grad:
                    yield f"{t.lora_name}.{name}", name, p, t.record.loraplus_tensors
        for n in self.norms:
            for name, p in self._norm_params(n):
                if p.requires_grad:
                    yield f"{n.lora_name}.{name}", name, p, ()

    def param_groups(
        self,
        lr: float,
        weight_decay: float = 0.0,
        rules: str = "moss",
        use_muon: bool = False,
        **kw,
    ) -> list[dict]:
        """Optimiser groups by weight-decay rule, Muon eligibility and the LoRA+ ratio
        (`lr_ratio_up`, default the configuration's) on each kind's zero-initialised
        factors."""
        ratio = float(kw.get("lr_ratio_up", self.config.lr_ratio_up))
        entries = list(self._entries())
        return groups.build(entries, lr, weight_decay, rules, use_muon, ratio)

    def state_dict(self) -> dict[str, Tensor]:
        """Raw trainable state (exact resume): every parameter and buffer by name."""
        return {k: v.detach().clone() for k, v in self._live().items()}

    def _live(self) -> dict[str, Tensor]:
        live = {}
        for t in self.targets:
            for name, v in nn.Module.state_dict(t.lyco, keep_vars=True).items():
                live[f"{t.lora_name}.{name}"] = v
        for n in self.norms:
            for name, v in self._norm_params(n):
                live[f"{n.lora_name}.{name}"] = v
        return live

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Copy a `state_dict()` back in place; the keys must match exactly."""
        live = self._live()
        unknown = sorted(set(state) - set(live))
        missing = sorted(set(live) - set(state))
        if unknown or missing:
            raise KeyError(
                f"adapter state mismatch: {len(unknown)} unknown keys "
                f"{unknown[:4]}, {len(missing)} missing keys {missing[:4]}"
            )
        with torch.no_grad():
            for key, value in state.items():
                live[key].copy_(value.to(live[key].device, live[key].dtype))
        self.bind()

    def tensors(
        self,
        fmt: str,
        style: str = "lora_unet",
        row_scales=None,
        fused_qkv: bool = False,
        base_of: Callable[[Target], Tensor | None] | None = None,
    ) -> dict:
        """Adapter tensors in `fmt` ('lycoris' | 'comfy') under key `style`, evaluated
        outside training; `row_scales` (a folded model's per-row fold scales) divides
        each layer's dW rows; `fused_qkv` (comfy) writes each block's joint q / k / v
        as one `proj.qkv` layer; `base_of(target)` is the base weight a layer reads in
        place of its own (None: its own)."""
        out, joint = {}, {}
        member = {}
        if fused_qkv:
            for fused, parts in qkv_groups(t.path for t in self.targets).items():
                member.update({path: (fused, name) for name, path in parts.items()})
        with torch.no_grad(), evaluated(self.targets):
            for t in self.targets:
                inv = None
                if row_scales and t.path in row_scales:
                    inv = convert.reciprocal(row_scales[t.path])
                with _based(t, base_of):
                    per = convert.native(t) if fmt == "lycoris" else convert.comfy(t)
                    if inv is not None:
                        per = convert.apply_row_scale(t, per, inv)
                if t.path in member:
                    fused, name = member[t.path]
                    joint.setdefault(fused, {})[name] = (t, per, inv)
                    continue
                stem = (
                    t.lora_name if fmt == "lycoris" else files.layer_key(t.path, style)
                )
                out.update({f"{stem}.{k}": v for k, v in per.items()})
            for fused, parts in joint.items():
                stem = files.layer_key(fused, style)
                per = self._fused(parts, base_of)
                out.update({f"{stem}.{k}": v for k, v in per.items()})
            for n in self.norms:
                stem = (
                    n.lora_name if fmt == "lycoris" else files.layer_key(n.path, style)
                )
                out.update(
                    {f"{stem}.{k}": v for k, v in convert.norm_native(n).items()}
                )
        return out

    @staticmethod
    def _fused(parts: dict, base_of) -> dict[str, Tensor]:
        """One block's q / k / v sets {name: (target, tensors, inverse row scale)} as
        one fused layer: a LoRA when every set is one, else the exact `diff`."""
        rows = {name: t.linear.out_features for name, (t, _, _) in parts.items()}
        for name in ("q", "k", "v"):
            rows.setdefault(name, next(iter(rows.values())))
        if all(is_lora(per) for _, per, _ in parts.values()):
            return fused_lora({n: per for n, (_, per, _) in parts.items()}, rows)
        dense = {}
        for name, (t, _, inv) in parts.items():
            with _based(t, base_of):
                dense[name] = convert.scaled_diff(t, 1.0 if inv is None else inv)
        cols = next(iter(parts.values()))[0].linear.in_features
        return fused_diff(dense, rows, cols)

    def metadata(
        self, fmt: str, style: str, extra: dict | None = None
    ) -> dict[str, str]:
        config = self.config
        meta = {
            "kohakumoss_format": fmt,
            "kohakumoss_key_style": style,
            "kohakumoss_prefix": config.prefix,
            "kohakumoss_algo": config.algo,
            "ss_network_module": "lycoris.kohya",
            "ss_network_dim": str(config.dim),
            "ss_network_alpha": str(config.alpha),
            "ss_network_args": json.dumps(
                {"algo": config.lycoris_algo, **config.module_kwargs()},
                default=str,
            ),
        }
        meta.update({k: str(v) for k, v in (extra or {}).items()})
        return meta

    def merged_state(
        self, dtype: torch.dtype | None = torch.bfloat16, fused_qkv: bool = False
    ) -> dict[str, Tensor]:
        """The backbone's export state with every dW merged at the set's multiplier
        (fold undone by the backbone's own export)."""
        with evaluated(self.targets):
            return merged_state_dict(
                self.model,
                self.targets,
                self.norms,
                self.multiplier,
                dtype,
                fused_qkv,
            )

    def export(
        self,
        path: str,
        fmt: str,
        metadata: dict[str, str] | None = None,
        style: str = "lora_unet",
        dtype: torch.dtype | None = torch.bfloat16,
        row_scales: dict | None = None,
        fused_qkv: bool = False,
    ) -> None:
        """Write the adapters (lycoris / comfy; `row_scales` undoes a fold), a merged
        checkpoint (merged) or a diffusers transformer folder (diffusers); the merged
        formats need the backbone and undo its fold themselves. `fused_qkv` writes the
        ComfyUI fused q/k/v layout (comfy, merged)."""
        if fmt not in FORMATS:
            raise KeyError(f"export format {fmt!r} not in {FORMATS}")
        if fused_qkv and fmt in ("lycoris", "diffusers"):
            raise ValueError(f"fused_qkv applies to 'comfy' and 'merged', not {fmt!r}")
        if fmt in ("lycoris", "comfy"):
            raw = self.tensors(fmt, style, row_scales, fused_qkv)
            meta = self.metadata(fmt, style, metadata)
            if fused_qkv:
                meta[FUSED_KEY] = "1"
            files.write(path, files.to_storage(raw, dtype), meta)
            return
        if fmt == "merged":
            state = self.merged_state(dtype, fused_qkv)
            meta = {"kohakumoss_format": "merged", **(metadata or {})}
            files.write(path, state, meta)
            return
        write_diffusers(self.merged_state(dtype), path, self.model.cfg)

    def state_metadata(self) -> dict[str, str]:
        """The metadata of a `state_dict()` file: RAW_TAG, the configuration under
        SPEC_KEY and the weight fold the layers trained on (FOLD_KEY)."""
        return {
            RAW_TAG: "1",
            SPEC_KEY: json.dumps(asdict(self.config)),
            **fold_metadata(self.model),
        }

    def save_state(self, path: str) -> None:
        """Raw state + `state_metadata()` as a safetensors file (exact reload)."""
        meta = self.state_metadata()
        files.write(path, files.to_storage(self.state_dict(), None), meta)

    def merge_into(self, multiplier: float = 1.0) -> None:
        """Bake every dW (and norm delta) at `multiplier` into its base weight and
        detach the adapters."""
        stored = [t.path for t in self.targets if t.stored]
        if stored:
            raise RuntimeError(
                f"cannot merge into quantized/streamed base storage: {stored[:4]}; "
                "export a merged checkpoint instead"
            )
        with torch.no_grad(), evaluated(self.targets):
            for t in self.targets:
                dw, db = t.delta(multiplier)
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
                for _, d in n.deltas():
                    d.multiplier = multiplier
                norm_delta.detach(n.module, True, "weight")
                if n.bias_delta is not None:
                    norm_delta.detach(n.module, True, "bias")
        self.targets, self.norms = [], []
        rebind_owner(self.model)

    def remove(self) -> None:
        """Detach every adapter; the base layers forward as before apply()."""
        for t in self.targets:
            self._detach(t)
        for n in self.norms:
            norm_delta.detach(n.module, False, "weight")
            if n.bias_delta is not None:
                norm_delta.detach(n.module, False, "bias")
        self.targets, self.norms = [], []
        rebind_owner(self.model)

    @staticmethod
    def _detach(t: Target) -> None:
        vars(t.linear).pop("forward", None)
        vars(t.linear).pop("train", None)
        if CHILD in t.linear._modules:
            del t.linear._modules[CHILD]
