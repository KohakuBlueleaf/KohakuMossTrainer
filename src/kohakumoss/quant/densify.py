"""Stored quantized bases back to dense weights, and the format a model's stored bases
carry: what a run whose precision differs from its checkpoint's stored one (e.g.
BASE_PRECISION="compute" on an int8 file) starts from. See
docs/internals/quant-kernels.md#stored-formats."""

import torch
from torch import nn

from kohakumoss.model.linear import MossLinear
from kohakumoss.quant.compute import ComputeWeight
from kohakumoss.quant.convert import QuantLinear
from kohakumoss.quant.embedding import QuantEmbedding
from kohakumoss.quant.w4a8 import W4A8Weight
from kohakumoss.quant.weights import QuantWeight

STORED = (QuantWeight, ComputeWeight, W4A8Weight)


def base_format(base) -> str:
    """The precision name of a stored base: `int8` / `fp8` / `w4a8`, or the fp8-compute
    precision of a ComputeWeight (`mxfp8` for the mx recipe, else `fp8_compute`)."""
    if isinstance(base, ComputeWeight):
        return "mxfp8" if base.recipe == "mx" else "fp8_compute"
    return base.fmt


def stored_bases(model: nn.Module) -> dict[str, MossLinear]:
    """{path: MossLinear} of every linear whose base is a stored quantized one."""
    return {
        path: m
        for path, m in model.named_modules()
        if isinstance(m, MossLinear) and isinstance(m.base, STORED)
    }


def stored_formats(model: nn.Module) -> set[str]:
    """The formats of `model`'s stored bases (`base_format`)."""
    return {base_format(m.base) for m in stored_bases(model).values()}


def stored_precision_of(model: nn.Module) -> str:
    """The one stored format of `model`'s bases: `compute` without any, `mixed` for
    several."""
    formats = stored_formats(model)
    if not formats:
        return "compute"
    return formats.pop() if len(formats) == 1 else "mixed"


def _meta(t):
    return None if t is None else torch.empty_like(t, device="meta")


def meta_base(base):
    """A copy of a stored base whose every tensor is an empty meta tensor of the same
    shape and dtype (format, rotation, group and recipe kept)."""
    if isinstance(base, QuantWeight):
        return QuantWeight(_meta(base.q), _meta(base.scale), base.fmt, base.rotation)
    if isinstance(base, W4A8Weight):
        return W4A8Weight(
            _meta(base.q),
            _meta(base.s_rel),
            _meta(base.s_channel),
            _meta(base.codebook),
            base.group,
            base.rotation,
            _meta(base.correction),
        )
    return ComputeWeight(base.recipe, {n: _meta(t) for n, t in base.tensors().items()})


def mirror_stored(twin: nn.Module, real: nn.Module) -> int:
    """Give every linear of `twin` whose namesake in `real` holds a stored base a meta
    copy of that base (`meta_base`); returns how many."""
    linears = stored_bases(real)
    for path, lin in linears.items():
        target = twin.get_submodule(path)
        dtype = target.weight_like.dtype
        target.weight = nn.Parameter(
            torch.empty(0, dtype=dtype, device="meta"), requires_grad=False
        )
        target.set_base(meta_base(lin.base))
    return len(linears)


@torch.no_grad()
def densify_bases(model: nn.Module) -> int:
    """Replace every stored base by its dense weight (the base's dequantized W, in the
    dtype of the linear's placeholder parameter); returns how many."""
    linears = stored_bases(model)
    for lin in linears.values():
        dtype = lin.weight_like.dtype
        w = lin.base.weight(dtype)
        lin.clear_base()
        lin.weight = nn.Parameter(w, requires_grad=False)
    return len(linears)


@torch.no_grad()
def densify_text(model: nn.Module, dtype: torch.dtype) -> int:
    """Replace every QuantLinear / QuantEmbedding of a text model by nn.Linear /
    nn.Embedding holding its dequantized weight in `dtype`; returns how many."""
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, QuantLinear):
                dense = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device="meta",
                )
                dense.weight = nn.Parameter(child.qweight.weight(dtype), False)
                if child.bias is not None:
                    dense.bias = nn.Parameter(child.bias.detach().to(dtype), False)
            elif isinstance(child, QuantEmbedding):
                dense = nn.Embedding(
                    child.num_embeddings, child.embedding_dim, device="meta"
                )
                dense.weight = nn.Parameter(child.weight(dtype), False)
            else:
                continue
            setattr(parent, name, dense)
            count += 1
    return count
