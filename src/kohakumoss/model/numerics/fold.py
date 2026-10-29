"""Weight fold: scale the joint and text residual streams by constants, in the weights
(`via="gate"` uniform, optional pre-gate scale; `via="proj"` per block; eps exact or
stock). See docs/concepts/numerics.md."""

from collections.abc import Sequence

import torch
from torch import Tensor

from kohakumoss.model.backbone import MossBackbone


def _scales(model: MossBackbone, c: float | Sequence[float]) -> tuple[float, ...]:
    """Per-block scales of `c`: one factor (a number or a one-item sequence) for every
    block, or one per block."""
    depth = model.cfg.depth
    if isinstance(c, (int, float)):
        return (float(c),) * depth
    cs = tuple(float(v) for v in c)
    if len(cs) == 1:
        return cs * depth
    if len(cs) != depth:
        raise ValueError(f"per-layer fold needs {depth} scales, got {len(cs)}")
    return cs


def writer_scales(model: MossBackbone, c_in: float) -> dict[str, float]:
    """Embedder / text-refiner writer parameters -> c_in."""
    out = {}
    for name in ("video_embedder", "text_embedder.proj"):
        out[f"{name}.weight"] = c_in
        out[f"{name}.bias"] = c_in
    for i in range(len(model.text_embedder.blocks)):
        for lin in ("proj", "ffn_down"):
            out[f"text_embedder.blocks.{i}.{lin}.weight"] = c_in
            out[f"text_embedder.blocks.{i}.{lin}.bias"] = c_in
    return out


def _row_scales(
    model: MossBackbone,
    cs: tuple[float, ...],
    via: str,
    pregate: tuple[float, float] = (1.0, 1.0),
) -> dict[str, Tensor | float]:
    """Parameter name -> multiplier (scalar, or per-output-row tensor) of the fold."""
    out: dict[str, Tensor | float] = writer_scales(model, cs[0])
    dim = model.cfg.dim
    s_a, s_f = pregate
    match via:
        case "gate":
            if len(set(cs)) != 1:
                raise ValueError("via='gate' folds one uniform scale")
            rows = torch.ones(6 * dim, dtype=torch.float32)
            rows[2 * dim : 3 * dim] = cs[0] / s_a
            rows[5 * dim : 6 * dim] = cs[0] / s_f
            out["shared_block_modulation.weight"] = rows
            out["shared_block_modulation.bias"] = rows
            for i in range(model.cfg.depth):
                if s_a != 1.0:
                    out[f"layers.{i}.proj.o.weight"] = s_a
                if s_f != 1.0:
                    out[f"layers.{i}.proj.ffn_down.weight"] = s_f
        case "proj":
            if pregate != (1.0, 1.0):
                raise ValueError("pregate scaling applies to via='gate'")
            for i, c in enumerate(cs):
                out[f"layers.{i}.proj.o.weight"] = c
                out[f"layers.{i}.proj.ffn_down.weight"] = c
        case _:
            raise ValueError(f"unknown fold mode {via!r}")
    return out


def norm_eps_scales(
    model: MossBackbone, cs: tuple[float, ...], per_block: bool
) -> dict[str, float]:
    """Norm module path -> eps multiplier for streams scaled by cs."""
    c_in = cs[0]
    out = {}
    for i in range(len(model.text_embedder.blocks)):
        out[f"text_embedder.blocks.{i}.norm1"] = c_in**2
        out[f"text_embedder.blocks.{i}.norm2"] = c_in**2
    prev = c_in
    for i, c in enumerate(cs):
        out[f"layers.{i}.norm1"] = (prev if per_block else c) ** 2
        out[f"layers.{i}.norm2"] = c**2
        prev = c
    out["final_norm_video"] = cs[-1] ** 2
    return out


def residual_ratios(cs: tuple[float, ...]) -> tuple[float, ...]:
    """Stream rescale applied at each block's first residual add: c_l / c_{l-1}."""
    return (1.0,) + tuple(b / a for a, b in zip(cs[:-1], cs[1:]))


def set_norm_eps(model: MossBackbone, mult: dict[str, float]) -> dict[str, float]:
    """Multiply each listed norm's eps; returns the previous values."""
    modules = dict(model.named_modules())
    before = {}
    for name, m in mult.items():
        before[name] = modules[name].eps
        modules[name].eps = modules[name].eps * m
    return before


def _multiply(t: Tensor, s: Tensor | float, inverse: bool) -> Tensor:
    """t * s (or t / s) in fp32 (or t's wider dtype), rounded once to t's dtype; exact
    for power-of-two s."""
    dt = torch.promote_types(t.dtype, torch.float32)
    if isinstance(s, Tensor):
        s = s.to(device=t.device, dtype=dt)
        s = s.reshape(-1, *([1] * (t.dim() - 1)))
    m = (1.0 / s) if inverse else s
    return (t.to(dt) * m).to(t.dtype)


def fold_rows(model: MossBackbone) -> dict[str, Tensor | float]:
    """Parameter name -> multiplier of the fold the model carries ({} when unfolded)."""
    if model.fold_scales is None:
        return {}
    return _row_scales(model, model.fold_scales, model.fold_via, model.fold_pregate)


def unscaled(t: Tensor, s: Tensor | float) -> Tensor:
    """`t` with the fold multiplier `s` undone (fp32 math, one rounding)."""
    return _multiply(t, s, inverse=True)


def _dense_targets(model: MossBackbone, names) -> None:
    """ValueError when a fold would scale a weight held as stored (quantized /
    streamed) base storage, whose dense parameter is a placeholder."""
    stored = []
    for name in names:
        path, _, _ = name.rpartition(".")
        if getattr(model.get_submodule(path), "base", None) is not None:
            stored.append(path)
    if stored:
        raise ValueError(
            f"the fold scales stored bases {stored[:3]} ({len(stored)}): run the "
            "16-bit weights (BASE_PRECISION='compute'), or fold without pre-gate scales"
        )


def _record(model, cs, via, pregate, eps_mode, base_eps) -> None:
    model.fold_scales = cs
    model.fold_via = via
    model.fold_pregate = pregate
    model.fold_eps_mode = eps_mode
    model._fold_eps = base_eps


@torch.no_grad()
def fold(
    model: MossBackbone,
    c: float | Sequence[float],
    via: str | None = None,
    pregate: tuple[float, float] = (1.0, 1.0),
    eps: str = "exact",
) -> None:
    """Fold scale(s) into the weights in place; call before adapters / base swaps."""
    if model.fold_scales is not None:
        raise RuntimeError("model is already folded; unfold first")
    cs = _scales(model, c)
    via = via or ("gate" if len(set(cs)) == 1 else "proj")
    params = dict(model.named_parameters())
    rows = _row_scales(model, cs, via, pregate)
    _dense_targets(model, rows)
    for name, s in rows.items():
        params[name].copy_(_multiply(params[name], s, inverse=False))
    mult = norm_eps_scales(model, cs, per_block=via == "proj")
    modules = dict(model.named_modules())
    base_eps = {name: modules[name].eps for name in mult}
    if eps == "exact":
        set_norm_eps(model, mult)
    model.set_residual_scale(residual_ratios(cs) if via == "proj" else (1.0,) * len(cs))
    _record(model, cs, via, pregate, eps, base_eps)


def adopt_fold(
    model: MossBackbone,
    c: float,
    pregate: tuple[float, float] = (1.0, 1.0),
    eps: str = "stock",
) -> None:
    """Record that the loaded weights already carry a uniform gate fold (fp16-fix file);
    `eps="exact"` switches the norms to the exact eps * c^2."""
    cs = _scales(model, c)
    mult = norm_eps_scales(model, cs, per_block=False)
    modules = dict(model.named_modules())
    base_eps = {name: modules[name].eps for name in mult}
    if eps == "exact":
        set_norm_eps(model, mult)
    _record(model, cs, "gate", tuple(pregate), eps, base_eps)


@torch.no_grad()
def unfold(model: MossBackbone) -> None:
    """Undo `fold` / `adopt_fold` in place (eps restored exactly)."""
    if model.fold_scales is None:
        return
    params = dict(model.named_parameters())
    rows = fold_rows(model)
    _dense_targets(model, rows)
    for name, s in rows.items():
        params[name].copy_(unscaled(params[name], s))
    modules = dict(model.named_modules())
    for name, eps in model._fold_eps.items():
        modules[name].eps = eps
    model.set_residual_scale((1.0,) * model.cfg.depth)
    model.fold_scales = None


def unfold_state(model: MossBackbone, sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """`sd` (keys and space of this model's folded weights) with the fold undone;
    affected tensors are copies."""
    out = dict(sd)
    for name, s in fold_rows(model).items():
        out[name] = unscaled(out[name], s)
    return out


def _output_scales(
    model: MossBackbone, rows: dict[str, Tensor | float]
) -> dict[str, Tensor]:
    out = {}
    for name, s in rows.items():
        if not name.endswith(".weight"):
            continue
        module = name[: -len(".weight")]
        n = model.get_submodule(module).out_features
        out[module] = s.float() if isinstance(s, Tensor) else torch.full((n,), float(s))
    return out


def output_row_scales(model: MossBackbone) -> dict[str, Tensor]:
    """Module path -> fp32 per-output-row scale the fold applied to that linear's output;
    exporters divide an adapter's delta W rows by it."""
    return _output_scales(model, fold_rows(model))


def fold_record(model: MossBackbone) -> dict | None:
    """The weight fold `model` carries as a JSON record (None when unfolded): per-block
    `scales`, `via`, the `pregate` (s_attn, s_ffn) and the norm `eps` mode."""
    if model.fold_scales is None:
        return None
    return {
        "scales": [float(c) for c in model.fold_scales],
        "via": model.fold_via,
        "pregate": [float(s) for s in model.fold_pregate],
        "eps": model.fold_eps_mode,
    }


def _floats(values) -> list[float]:
    """`values` as floats; a uniform sequence as its one factor."""
    out = [float(v) for v in values]
    return out[:1] if len(set(out)) == 1 else out


def same_fold(a: dict | None, b: dict | None) -> bool:
    """Two fold records scale the same weight rows by the same factors (eps aside)."""
    if a is None or b is None:
        return a is b
    return (
        a["via"] == b["via"]
        and _floats(a["scales"]) == _floats(b["scales"])
        and [float(s) for s in a["pregate"]] == [float(s) for s in b["pregate"]]
    )


def record_row_scales(model: MossBackbone, record: dict | None) -> dict[str, Tensor]:
    """`output_row_scales` of the fold `record` describes, on `model`'s architecture
    ({} for None), whatever fold `model` itself carries."""
    if record is None:
        return {}
    cs = _scales(model, record["scales"])
    pregate = tuple(float(s) for s in record["pregate"])
    return _output_scales(model, _row_scales(model, cs, record["via"], pregate))
