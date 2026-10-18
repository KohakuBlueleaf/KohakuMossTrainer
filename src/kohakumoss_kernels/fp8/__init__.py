"""fp8 / MXFP8 compute over pre-quantized frozen weights: formats (the eager reference
quantizers) and the `fp8_linear` op. See docs/internals/quant-kernels.md."""

from kohakumoss_kernels.fp8.formats import (
    BLOCK,
    E4M3,
    E4M3_MAX,
    MX_BLOCK,
    RECIPES,
    dequantize,
    quantize,
    swizzle_mx,
)
from kohakumoss_kernels.fp8.op import (
    dgrad_reference,
    fwd_reference,
    make_fp8_linear,
    register_launchers,
)
from kohakumoss_kernels.fp8.train import make_fp8_linear_train

__all__ = [
    "BLOCK",
    "E4M3",
    "E4M3_MAX",
    "MX_BLOCK",
    "RECIPES",
    "dequantize",
    "dgrad_reference",
    "fwd_reference",
    "make_fp8_linear",
    "make_fp8_linear_train",
    "quantize",
    "register_launchers",
    "swizzle_mx",
]
