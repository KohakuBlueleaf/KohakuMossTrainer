"""Triton backend of `fp8_linear` (CUDA sm89+, Intel XPU): the GEMM paths of the form
c = Q(a) B^T (forward: a=x, B=W; fp8 input gradient: a=dy, B=W^T, the dgrad copy),
timed per problem by the tuner and bound once per settled key. Each path includes its
activation quantizer; the paths per recipe are in docs/internals/quant-kernels.md."""

import functools

import torch
import torch.nn.functional as F

from kohakumoss_common.log import get_logger
from kohakumoss_kernels import compile_setup, registry, tuning
from kohakumoss_kernels.backends.triton.fp8.fused import FUSED
from kohakumoss_kernels.backends.triton.fp8.gemm import (
    dot_scaled_mm,
    promote_mm,
    scaled_mm,
)
from kohakumoss_kernels.backends.triton.fp8.quantize import (
    quantize_act,
    quantize_blocks,
    quantize_mx_swizzled,
)
from kohakumoss_kernels.constraints import requires
from kohakumoss_kernels.errors import BackendUnavailable, describe, is_build_error
from kohakumoss_kernels.fp8.formats import BLOCK, MX_BLOCK, RECIPES, dequantize
from kohakumoss_kernels.fp8.op import (
    make_fp8_linear,
    make_fp8_linear_infer,
    register_launchers,
)
from kohakumoss_kernels.fp8.train import make_fp8_linear_train
from kohakumoss_kernels.hardware.arch import arch_of
from kohakumoss_kernels.tuning import tuner
from kohakumoss_kernels.tuning.buckets import pow2

log = get_logger(__name__)
_TILE_KEYS = ("bm", "bn", "bk", "warps", "stages")
SCALED_TILES = (
    (128, 128, 128, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 128, 64, 4, 4),
    (64, 128, 256, 4, 3),
    (64, 128, 128, 4, 4),
    (128, 64, 128, 4, 4),
    (64, 64, 128, 4, 4),
    (32, 64, 128, 4, 4),
    (16, 64, 128, 4, 4),
)
DOT_SCALED_TILES = (
    (64, 128, 256, 4, 3),
    (128, 64, 256, 4, 3),
    (128, 128, 128, 8, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 8, 3),
    (64, 64, 128, 4, 4),
    (32, 64, 128, 4, 4),
)
PROMOTE_TILES = (
    (128, 128, 0, 8, 4),
    (128, 256, 0, 8, 3),
    (128, 128, 0, 4, 4),
    (64, 128, 0, 4, 4),
    (128, 64, 0, 4, 4),
    (64, 64, 0, 4, 4),
    (32, 64, 0, 4, 4),
)
FUSED_TILES = (
    (64, 128, 128, 4, 3),
    (32, 128, 128, 4, 3),
    (64, 64, 128, 4, 4),
    (32, 64, 128, 4, 4),
    (16, 64, 128, 4, 4),
)
MX_TILE_K = 4 * MX_BLOCK
W16 = ({"gemm": "w16", "impl": "eager"}, {"gemm": "w16", "impl": "compile"})
VENDOR = {"gemm": "vendor"}


def _tiles(gemm: str, table: tuple, m: int, **extra) -> list[dict]:
    limit = max(pow2(m, 16), min(t[0] for t in table))
    fits = [t for t in table if t[0] <= limit]
    ranked = sorted(fits, key=lambda t: -(t[0] * t[1]) if m >= 1024 else t[0])
    return [{"gemm": gemm, **dict(zip(_TILE_KEYS, t)), **extra} for t in ranked]


def candidates(
    recipe: str, sm: int, m: int, fp8: bool = True, k: int | None = None
) -> list[dict]:
    """Every GEMM path of `recipe` on `sm` for M rows, the planner's first pick first;
    `fp8=False` (one fp8 weight copy) leaves only the 16-bit paths. With the contraction
    length `k`, mx paths that read whole 128-column tiles are left out when
    `k % 128`."""
    if not fp8:
        return list(W16)
    fused = _tiles("fused", FUSED_TILES, m, acc="promote")
    triton = _tiles("triton", SCALED_TILES, m, acc="promote")
    match recipe:
        case "tensor":
            gemms = [VENDOR, *triton]
        case "row" if sm >= 90:
            gemms = [VENDOR, *triton, *fused]
        case "row":
            gemms = [*triton, VENDOR, *fused]
        case "block":
            gemms = [*_tiles("promote", PROMOTE_TILES, m), *fused]
        case "mx" if sm >= 100:
            whole = k is None or k % MX_TILE_K == 0
            gemms = [VENDOR] if whole else []
            gemms += _tiles("dot_scaled", DOT_SCALED_TILES, m)
            gemms += _tiles("promote", PROMOTE_TILES, m)
            gemms += fused if whole else []
        case "mx":
            gemms = _tiles("promote", PROMOTE_TILES, m, acc="upcast")
        case _:
            raise ValueError(f"fp8 recipe {recipe!r} not in {RECIPES}")
    return [*gemms, *W16]


def _vendor(recipe, a, bq, bs, baux, bias):
    b = None if bias is None else bias.to(a.dtype)
    match recipe:
        case "tensor":
            aq, sa = quantize_act(a, "tensor")
            return torch._scaled_mm(
                aq, bq.t(), scale_a=sa.reshape(()), scale_b=bs.reshape(()),
                bias=b, out_dtype=a.dtype,
            )  # fmt: skip
        case "row":
            aq, sa = quantize_act(a, "row")
            return torch._scaled_mm(
                aq, bq.t(), scale_a=sa[:, None], scale_b=bs[None, :],
                bias=b, out_dtype=a.dtype,
            )  # fmt: skip
        case "mx":
            aq, sa = quantize_mx_swizzled(a)
            e8m0, mx = torch.float8_e8m0fnu, torch._C._ScalingType.BlockWise1x32
            swz = torch._C._SwizzleType.SWIZZLE_32_4_4
            return F.scaled_mm(
                aq, bq.t(), sa.view(e8m0), mx, baux.view(e8m0), mx, swz, swz,
                bias=b, output_dtype=a.dtype,
            )  # fmt: skip
    raise ValueError(f"no vendor GEMM for fp8 recipe {recipe!r}")


def _w16_fwd(recipe, x, q, scale, bias):
    b = None if bias is None else bias.to(x.dtype)
    return F.linear(x, dequantize(q, scale, recipe, x.dtype), b)


def _w16_dgrad(recipe, dy, q, scale):
    return dy @ dequantize(q, scale, recipe, dy.dtype)


@functools.cache
def _compiled(fn):
    compile_setup.apply()
    return torch.compile(fn, dynamic=True)


def gemm_runner(cfg: dict, recipe: str):
    """(a16 [M, K], bq [N, K], bs, baux, bias) -> Q(a) bq^T in a's dtype, one fp8 path."""
    match cfg["gemm"]:
        case "vendor":
            return functools.partial(_vendor, recipe)
        case "triton":

            def run(a, bq, bs, baux, bias):
                aq, sa = quantize_act(a, recipe)
                return scaled_mm(aq, bq, sa, bs, bias, a.dtype, cfg)

        case "promote":
            sub = MX_BLOCK if recipe == "mx" else BLOCK
            b_rows = 1 if recipe == "mx" else BLOCK
            tile = {**cfg, "bk": sub}

            def run(a, bq, bs, baux, bias):
                aq, sa = quantize_blocks(a, mx=recipe == "mx")
                return promote_mm(aq, bq, sa, bs, bias, a.dtype, tile, sub, b_rows)

        case "dot_scaled":

            def run(a, bq, bs, baux, bias):
                aq, sa = quantize_blocks(a, mx=True)
                return dot_scaled_mm(aq, sa, bq, bs, bias, a.dtype, cfg)

        case "fused":
            mm = FUSED[recipe]

            def run(a, bq, bs, baux, bias):
                return mm(a, bq, bs, bias, cfg)

        case gemm:
            raise ValueError(f"unknown fp8 GEMM path {gemm!r}")
    return run


def _fwd_runner(cfg, recipe):
    if cfg["gemm"] == "w16":
        fn = functools.partial(_w16_fwd, recipe)
        fn = fn if cfg["impl"] == "eager" else _compiled(fn)
        return lambda x, q, scale, aux, bias: fn(x, q, scale, bias)
    return gemm_runner(cfg, recipe)


def _dgrad_runner(cfg, recipe):
    if cfg["gemm"] == "w16":
        fn = functools.partial(_w16_dgrad, recipe)
        fn = fn if cfg["impl"] == "eager" else _compiled(fn)
        return lambda dy, q, scale, q_t, scale_t, aux_t: fn(dy, q, scale)
    gemm = gemm_runner(cfg, recipe)
    return lambda dy, q, scale, q_t, scale_t, aux_t: gemm(dy, q_t, scale_t, aux_t, None)


_RESOLVED: dict[tuple, object] = {}
_DEAD: dict[tuple, str] = {}
tuner.on_reset(_RESOLVED.clear)


def _config_key(a, kind: str, recipe: str, cfg: dict) -> tuple:
    """Availability key of one config (a tile that cannot build leaves the others)."""
    return (a.device.index, a.dtype, kind, recipe, tuple(sorted(cfg.items())))


def _probe(cfg, recipe, runner, args, a, kind) -> bool:
    """Run config `cfg` once on the call's own inputs under no_grad; a config that
    cannot run on this device (build / unsupported-feature errors) is marked dead and
    logged once. Out-of-memory and other errors propagate."""
    key = _config_key(a, kind, recipe, cfg)
    if key in _DEAD:
        return False
    try:
        with torch.no_grad():
            runner(cfg, recipe)(*args)
    except Exception as err:  # noqa: BLE001
        if not is_build_error(err):
            raise
        _DEAD[key] = describe(err)
        log.warning(
            f"fp8_{kind} path {cfg['gemm']!r} unavailable on {a.device} "
            f"({recipe}, {a.dtype}); next path",
            error=_DEAD[key][:160],
        )
        return False
    return True


def _table_key(kind, m, n, k, a, recipe, bias, fp8) -> str:
    b = "x" if bias is None else "b"
    return f"{kind}_m{pow2(m, 16)}_n{n}_k{k}_{a.dtype}_{recipe}_{b}_f{int(fp8)}"


def _select(kind, a, n, k, recipe, bias, fp8, runner, args, in_backward):
    """The bound runner of this problem (`k` the contraction length): the tuner's pick
    among the live configs, probed before it binds."""
    m = a.shape[0]
    key = (kind, a.device.index, pow2(m, 16), n, k, a.dtype, recipe, bias is None, fp8)  # fmt: skip
    run = _RESOLVED.get(key)
    if run is not None:
        return run
    table_key = _table_key(kind, m, n, k, a, recipe, bias, fp8)
    sm = arch_of(a.device).sm
    ranked = candidates(recipe, sm, m, fp8, k)
    ident = (a.device.index, a.dtype, kind, recipe)
    while True:
        alive = [c for c in ranked if _config_key(a, kind, recipe, c) not in _DEAD]
        if not alive:
            errors = {str(d[4]): v for d, v in _DEAD.items() if d[:4] == ident}
            raise BackendUnavailable(f"fp8_{kind}", errors, "no fp8 path runs here")
        cfg = tuning.select(
            f"fp8_{kind}",
            a.device,
            table_key,
            lambda: alive,
            lambda c: lambda: runner(c, recipe)(*args),
            shortlist=len(alive),
            in_backward=in_backward,
        )
        if _probe(cfg, recipe, runner, args, a, kind):
            break
        tuning.forget(f"fp8_{kind}", a.device, table_key)
    run = runner(cfg, recipe)
    if tuning.decided(f"fp8_{kind}", a.device, table_key):
        _RESOLVED[key] = run
    return run


def fwd(x, q, scale, aux, bias, recipe):
    args = (x, q, scale, aux, bias)
    n, k = q.shape
    run = _select("fwd", x, n, k, recipe, bias, True, _fwd_runner, args, False)
    return run(*args)


def dgrad(dy, q, scale, q_t, scale_t, aux_t, recipe):
    args = (dy, q, scale, q_t, scale_t, aux_t)
    n, k = q.shape
    fp8 = q_t is not None
    run = _select("dgrad", dy, k, n, recipe, None, fp8, _dgrad_runner, args, True)
    return run(*args)


def _known_recipe(query):
    recipe = query.get("recipe", "row")
    return None if recipe in RECIPES else f"unknown fp8 recipe {recipe!r}"


register_launchers("triton", fwd, dgrad)
for _op, _make in (
    ("fp8_linear", make_fp8_linear),
    ("fp8_linear_infer", make_fp8_linear_infer),
    ("fp8_linear_train", make_fp8_linear_train),
):
    registry.register(
        _op,
        "triton",
        _make("triton"),
        supports=requires(kinds=("cuda", "xpu"), min_sm=89, extra=_known_recipe),
    )
