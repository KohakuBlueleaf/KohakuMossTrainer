"""Choose each layer's bound forward once: fused LyCORIS kernel or eager body.

Every decision here happens at bind time and yields a distinct callable; the bound
forward never inspects mode, multiplier, flags or training state.
"""

from functools import partial

import torch
from torch import nn

from kohakumoss.adapters import delta as dl
from kohakumoss.adapters.backend import is_fused
from kohakumoss.adapters.bind import eager, fused, kits
from kohakumoss.adapters.target import Target


def has_scalar_param(lyco) -> bool:
    scalar = getattr(lyco, "scalar", None)
    return isinstance(scalar, torch.Tensor) and scalar.requires_grad


def dropping(lyco) -> bool:
    """Input or rank dropout configured on this module (fused kernels carry neither)."""
    drop = getattr(lyco, "dropout", 0.0)
    input_drop = (
        isinstance(drop, nn.Dropout) if isinstance(drop, nn.Module) else drop > 0
    )
    return input_drop or getattr(lyco, "rank_dropout", 0.0) > 0


def _lin(t: Target):
    return (
        fused.lin_nobias if t.linear.bias is None else partial(fused.lin_bias, t.linear)
    )


def _scale(m: float):
    return fused.same if m == 1.0 else partial(fused.times, m)


def _mix(m: float):
    return fused.keep if m == 1.0 else partial(fused.interpolate, m)


def _scalared(lyco, unit):
    return partial(fused.with_scalar, lyco, unit) if has_scalar_param(lyco) else unit


def _unit(t: Target, kit):
    """Fused merge-kernel ΔW builder (base -> ΔW at multiplier 1), or None."""
    lyco = t.lyco
    match t.kind:
        case "locon":
            unit = kit.lora_unit(lyco)
        case "loha":
            unit = kit.loha_unit(lyco)
        case "lokr" if not lyco.tucker:
            unit = kit.lokr_unit(lyco)
        case "glora":
            unit = partial(fused.glora_unit, lyco, kit.backend)
        case _:
            unit = None
    return None if unit is None else _scalared(lyco, unit)


def _added(t: Target, diff):
    lyco = t.lyco
    if diff is None:
        return None
    if has_scalar_param(lyco):
        return partial(fused.add_base_scalar, t.linear, lyco, diff)
    return partial(fused.add_base, t.linear, diff)


def _fused_bypass(t: Target, kit, m: float):
    lyco = t.lyco
    match t.kind:
        case "locon":
            return _added(t, kit.lora(lyco, lyco.scale * m))
        case "loha":
            return _added(t, kit.loha(lyco, lyco.scale * m))
        case "lokr" if not lyco.tucker:
            return _added(t, kit.lokr(lyco, lyco.scale * m))
        case "oft" if m == 1.0:
            return kit.oft(t.linear, lyco)
        case "boft":
            return kit.boft(t.linear, lyco, m)
        case "ia3":
            fn = fused.ia3_bypass_in if lyco.train_input else fused.ia3_bypass_out
            return partial(fn, t.linear, lyco, m, kit.backend)
    return None


def _merged(t: Target, unit, m: float):
    if unit is None:
        return None
    return partial(fused.merged_forward, t.base_fn, unit, _scale(m), _lin(t))


def _fused_merge(t: Target, kit, m: float):
    lyco = t.lyco
    match t.kind:
        case "oft" if m == 1.0:
            return _merged(t, kit.oft_unit(lyco), 1.0)
        case "boft":
            return _merged(t, kit.boft_unit(lyco, m), 1.0)
        case "ia3":
            return _merged(t, partial(fused.ia3_unit, lyco, m, kit.backend), 1.0)
    return _merged(t, _unit(t, kit), m)


def _dora(t: Target, kit, m: float, use_fused: bool, detach: bool):
    unit = _unit(t, kit) if use_fused and not detach else None
    if unit is not None:
        t.fused = True
        return partial(
            fused.dora_forward,
            t.base_fn,
            t.lyco,
            unit,
            kit.decompose(t.lyco, t.wd_on_out),
            _mix(m),
            _lin(t),
        )
    if detach:
        fn = dl.detached_out if t.wd_on_out else dl.detached_in
        decompose = partial(dl.detached, fn, t.lyco)
    else:
        decompose = partial(dl.dora_merged, t)
    return partial(eager.eager_dora, t.base_fn, t.unit_fn, decompose, _mix(m), _lin(t))


def _eager(t: Target, mode: str, m: float):
    lyco = t.lyco
    if mode == "merge":
        return partial(
            eager.eager_merged, t.base_fn, lyco.shape, dl.bind_delta(t, m), _lin(t)
        )
    match t.kind:
        case "oft" if m != 1.0:
            return partial(
                eager.delta_bypass, t.linear, lyco.shape, dl.bind_delta(t, m)
            )
        case "ia3":
            fn = eager.ia3_in if lyco.train_input else eager.ia3_out
            return partial(fn, t.linear, lyco, m)
    return partial(eager.module_bypass, lyco, m)


def bind_target(
    t: Target,
    mode: str,
    backend: str,
    m: float,
    detach: bool,
    module_p: float,
    race: bool = False,
):
    """The forward to install on `t.linear` for this mode / backend / multiplier /
    training state.

    With `race` the fused ops are bound to LyCORIS's measured tier race
    (`bind.kits.Raced`), otherwise to `backend` directly. Quantized / streamed base
    storage cannot be merged into, so such layers bind the bypass form whatever
    `mode` says.
    """
    lyco = t.lyco
    lyco.multiplier = m
    mode = "bypass" if t.stored else mode
    training = lyco.training
    use_fused = is_fused(backend) and not dropping(lyco)
    t.fused = False
    match t.kind:
        case "full":
            return partial(eager.module_forward, lyco)
        case "dylora" if training:
            return partial(eager.dylora_train, t.linear, lyco, m)
        case "dylora":
            return partial(
                eager.delta_bypass, t.linear, lyco.shape, dl.bind_delta(t, m)
            )
    kit = kits.kit(backend, race)
    if t.wd:
        fn = _dora(t, kit, m, use_fused, detach)
    else:
        fn = None
        if use_fused:
            pick = _fused_bypass if mode == "bypass" else _fused_merge
            fn = pick(t, kit, m)
        t.fused = fn is not None
        fn = fn or _eager(t, mode, m)
    if module_p > 0 and training:
        fn = partial(eager.module_dropout, t.linear, module_p, fn)
    return fn


def training_dependent(t: Target, module_p: float) -> bool:
    """Whether the bound forward differs between train() and eval()."""
    return t.kind == "dylora" or module_p > 0
