"""The weightless twin of the run's backbone that the kernel prewarm records from: the
run's module structure, base storage kinds (int8 / fp8 / fp8_compute / mxfp8 bases, the
stored bases of a pre-quantized checkpoint with their rotation, fp8 compute caches),
parameter dtypes and grad flags, every tensor a stride-0 view of one zero on the
device. See docs/internals/tuning.md#prewarm."""

from functools import partial

import torch
from torch import Tensor, nn

from kohakumoss.adapters.build.target import CHILD
from kohakumoss.memory.apply import quantize_base, trainable_compute
from kohakumoss.model import resolve_config
from kohakumoss.model.build import build_shapes, weightless
from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.bind import rebind_quant, set_int8_compute
from kohakumoss.quant.compute import PARTS, ComputeCache
from kohakumoss.quant.densify import mirror_stored
from kohakumoss.training.knobs import is_full
from kohakumoss_common.weak import weak_call


def _storage(
    model, plan, compute: torch.dtype, full: bool, keep_fp32, real=None
) -> None:
    """The run's base storage swaps (training), on the meta device: the stored bases
    `real` loaded from its checkpoint (`mirror_stored`), then the plan's (int8 compute
    included)."""
    if real is not None:
        mirror_stored(model, real)
    quantize_base(model, plan, compute, True, full, keep_fp32)
    trainable_compute(model, plan, compute, True, full, keep_fp32)
    if plan.int8_compute:
        set_int8_compute(model, plan.int8_dgrad)


def _place_caches(model: nn.Module, device, compute: torch.dtype) -> None:
    """Every fp8 compute cache's meta parts -> `weightless` on `device`, bound there."""
    for m in model.modules():
        cache = getattr(m, "compute", None)
        if isinstance(cache, ComputeCache):
            for name in PARTS:
                t = getattr(cache, name)
                if t is not None and t.is_meta:
                    setattr(cache, name, weightless(t, device))
            cache.bind(compute, device)


def _grad_out(linear: MossLinear, zero: Tensor, x: Tensor) -> Tensor:
    return linear._impl(linear, x) + zero


def mirror(twin: nn.Module, real: nn.Module, device) -> None:
    """`real`'s run-time numerics on `twin`: norm eps, residual scales, fp16-fix mode
    and stream config; every linear `real` carries an adapter on gives an output that
    requires grad, as the adapter's does."""
    zero = torch.zeros((), device=device, requires_grad=True)
    sources = dict(real.named_modules())
    for name, m in twin.named_modules():
        src = sources.get(name)
        if src is None:
            continue
        eps = getattr(src, "eps", None)
        if isinstance(eps, float) and hasattr(m, "eps"):
            m.eps = eps
        if isinstance(m, MossLinear) and CHILD in src._modules:
            m.forward = weak_call(_grad_out, m, zero)
    twin.fp16_fix = real.fp16_fix
    twin.set_residual_scale(real.residual_scale)
    twin.set_stream_config(real.stream)


def run_shapes(k: dict, prec, plan, device: torch.device, real=None) -> nn.Module:
    """The weightless twin of the backbone `k` / `prec` / memory `plan` build on
    `device`; `real` (the built backbone, None for `kohakumoss-tune`-like use) gives
    the dtypes and grad flags by parameter name and the run-time numerics."""
    cfg = resolve_config(k["VARIANT"], k["ARCH_OVERRIDES"] or None)
    full = is_full(k)
    like = None
    if real is not None:
        like = {**dict(real.named_parameters()), **dict(real.named_buffers())}
    storage = partial(
        _storage,
        plan=plan,
        compute=prec.compute,
        full=full,
        keep_fp32=prec.keep_fp32,
        real=real,
    )
    twin = build_shapes(
        cfg,
        device,
        prec.compute,
        full,
        prec.keep_fp32,
        k["KERNELS"],
        prepare=storage,
        like=like,
    )
    _place_caches(twin, device, prec.compute)
    rebind_quant(twin, device, prec.compute, autograd=True)
    if real is not None:
        mirror(twin, real, device)
    return twin
