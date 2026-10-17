"""Choose each layer's bound forward once, from its kind record and the bind context.

Every decision here happens at bind time and yields a distinct callable; the bound
forward never inspects mode, multiplier, flags or training state.
"""

from functools import partial

from torch import nn

from kohakumoss.adapters.backend import is_fused
from kohakumoss.adapters.bind import compose, dora, eager, fused, kits
from kohakumoss.adapters.bind.context import BindCtx
from kohakumoss.adapters.target import Target


def dropping(lyco) -> bool:
    """Input or rank dropout configured on this module (fused kernels carry neither)."""
    drop = getattr(lyco, "dropout", 0.0)
    input_drop = (
        isinstance(drop, nn.Dropout) if isinstance(drop, nn.Module) else drop > 0
    )
    return input_drop or getattr(lyco, "rank_dropout", 0.0) > 0


def _merge(t: Target, ctx: BindCtx):
    rec = t.record
    if rec.merge is not None:
        return rec.merge(t, ctx)
    unit = compose.scalared(t.lyco, rec.fused_unit(t, ctx.kit))
    return compose.merged(t, unit, ctx.m)


def _eager(t: Target, ctx: BindCtx):
    if ctx.mode == "merge":
        lin = compose.lin_of(t)
        return partial(
            eager.eager_merged, t.base_fn, t.lyco.shape, t.delta_at(ctx.m), lin
        )
    if t.record.eager is not None:
        return t.record.eager(t, ctx)
    return partial(eager.module_bypass, t.lyco, ctx.m)


def _dora(t: Target, ctx: BindCtx):
    grad_free = not (ctx.detach and ctx.training)
    unit = None
    if ctx.use_fused and grad_free:
        unit = compose.scalared(t.lyco, t.record.fused_unit(t, ctx.kit))
    mix, lin = compose.mix_of(ctx.m), compose.lin_of(t)
    if unit is not None:
        t.fused = True
        decompose = ctx.kit.decompose(t.lyco, t.wd_on_out)
        return partial(fused.dora_forward, t.base_fn, t.lyco, unit, decompose, mix, lin)
    if ctx.detach and ctx.training:
        fn = dora.detached_out if t.wd_on_out else dora.detached_in
        decompose = partial(dora.detached, fn, t.lyco)
    else:
        decompose = partial(dora.decomposed, t, ctx.backend)
    return partial(eager.eager_dora, t.base_fn, t.unit_fn, decompose, mix, lin)


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
    training state; fused ops raced (`race`) or pinned to `backend`."""
    lyco = t.lyco
    lyco.multiplier = m
    lyco.kernel_backend = backend
    t.backend = backend
    training = lyco.training
    ctx = BindCtx(
        mode="bypass" if t.stored else mode,
        backend=backend,
        kit=kits.kit(backend, race),
        m=m,
        detach=detach,
        module_p=module_p,
        training=training,
        use_fused=is_fused(backend) and not (training and dropping(lyco)),
        draws=t.draws,
    )
    t.fused = False
    rec = t.record
    if rec.special is not None:
        return rec.special(t, ctx)
    if t.wd:
        fn = _dora(t, ctx)
    else:
        fn = None
        if ctx.use_fused:
            fn = rec.bypass(t, ctx) if ctx.mode == "bypass" else _merge(t, ctx)
        t.fused = fn is not None
        fn = fn or _eager(t, ctx)
    if module_p > 0 and training:
        fn = partial(eager.module_dropout, t.linear, t.draws, fn)
    return fn


def training_dependent(t: Target, module_p: float, detach: bool) -> bool:
    """Whether the bound forward differs between train() and eval()."""
    return (
        t.record.training_dependent
        or module_p > 0
        or dropping(t.lyco)
        or (t.wd and detach)
    )
