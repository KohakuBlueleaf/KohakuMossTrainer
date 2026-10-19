"""Choose each layer's bound forward once, from its kind record and the bind context.

Every decision here happens at bind time and yields a distinct callable; the bound
forward is that callable, called directly.
"""

from functools import partial

from torch import nn

from kohakumoss.adapters.bind import compose, dora, eager, forwards
from kohakumoss.adapters.bind.context import BindCtx
from kohakumoss.adapters.bind.kernel_ops import kernel_ops
from kohakumoss.adapters.build.target import Target


def dropping(lyco) -> bool:
    """Input or rank dropout configured on this module (the kernel ops carry neither)."""
    drop = getattr(lyco, "dropout", 0.0)
    input_drop = (
        isinstance(drop, nn.Dropout) if isinstance(drop, nn.Module) else drop > 0
    )
    return input_drop or getattr(lyco, "rank_dropout", 0.0) > 0


def _merge(t: Target, ctx: BindCtx):
    rec = t.record
    if rec.kernel_merge is not None:
        return rec.kernel_merge(t, ctx)
    unit = compose.with_scalar(t.lyco, rec.kernel_diff_weight(t, ctx.ops))
    if unit is None:
        return None
    merge = None
    if ctx.m == 1.0 and not compose.has_scalar_param(t.lyco):
        merge = rec.kernel_merged_weight(t, ctx.ops)
    return compose.merged(t, unit, ctx.m, merge)


def _eager(t: Target, ctx: BindCtx):
    if ctx.mode == "merge":
        lin = compose.lin_of(t)
        return eager.eager_merged(t.base_fn, t.lyco.shape, t.delta_at(ctx.m), lin)
    if t.record.module_forward is not None:
        return t.record.module_forward(t, ctx)
    return eager.module_bypass(t.layer, t.lyco, ctx.m)


def _dora_kernel(t: Target, ctx: BindCtx, mix, lin):
    """DoRA through the kernel ops (W + dW read inside the decompose op where it can),
    or None where the kind has no kernel dW."""
    unit = compose.with_scalar(t.lyco, t.record.kernel_diff_weight(t, ctx.ops))
    if unit is None:
        return None
    with_delta = ctx.ops.decompose_with_delta(t.lyco, t.wd_on_out)
    if with_delta is not None:
        return forwards.dora_with_delta(t.base_fn, t.lyco, unit, with_delta, mix, lin)
    decompose = ctx.ops.decompose(t.lyco, t.wd_on_out)
    if decompose is None:
        return None
    return forwards.dora(t.base_fn, t.lyco, unit, decompose, mix, lin)


def _stored_dora(t: Target):
    """The DoRA bypass of a stored base: input or output axis, with or without bias."""
    if not t.wd_on_out:
        return dora.bypass_in
    return dora.bypass_out if t.linear.bias is None else dora.bypass_out_biased


def _dora(t: Target, ctx: BindCtx):
    detach = ctx.detach and ctx.training
    if t.stored:
        fn = _stored_dora(t)
        return partial(fn, t.layer, t.base_fn, t.unit_fn, t.lyco, ctx.m, detach)
    mix, lin = compose.mix_of(ctx.m), compose.lin_of(t)
    if ctx.use_ops and not detach:
        fn = _dora_kernel(t, ctx, mix, lin)
        if fn is not None:
            return fn
    if detach:
        fn = dora.detached_out if t.wd_on_out else dora.detached_in
        decompose = dora.detached(fn, t.lyco)
    else:
        decompose = dora.decomposer(t, ctx.backend)
    return eager.eager_dora(t.base_fn, t.unit_fn, decompose, mix, lin)


def bind_target(
    t: Target,
    mode: str,
    backend: str,
    m: float,
    detach: bool,
    module_p: float,
    autotune: bool = False,
    fuse_base_add: bool = True,
    dynamic: str = "tokens",
):
    """The forward to install on `t.linear` for this mode / backend / multiplier /
    training state. Its kernel ops are autotuned (`autotune`) or fixed to `backend`,
    the base add inside the op where one exists (`fuse_base_add`); the autotuned ops
    are kept on `t.autotuned` for the prewarm."""
    lyco = t.lyco
    lyco.multiplier = m
    t.backend = backend
    training = lyco.training
    ops = kernel_ops(backend, autotune, fuse_base_add, dynamic)
    ctx = BindCtx(
        mode="bypass" if t.stored else mode,
        backend=backend,
        ops=ops,
        m=m,
        detach=detach,
        module_p=module_p,
        training=training,
        use_ops=not (training and dropping(lyco)),
        draws=t.draws,
    )
    rec = t.record
    if rec.own_forward is not None:
        fn = rec.own_forward(t, ctx)
    elif t.wd:
        fn = _dora(t, ctx)
    else:
        fn = None
        if ctx.use_ops:
            fn = rec.kernel_bypass(t, ctx) if ctx.mode == "bypass" else _merge(t, ctx)
        fn = fn or _eager(t, ctx)
    t.autotuned = ops.autotuned
    if rec.own_forward is None and module_p > 0 and training:
        fn = eager.module_dropout(t.layer, t.draws, fn)
    return fn


def training_dependent(t: Target, module_p: float, detach: bool) -> bool:
    """Whether the bound forward differs between train() and eval()."""
    return (
        t.record.training_dependent
        or module_p > 0
        or dropping(t.lyco)
        or (t.wd and detach)
    )
