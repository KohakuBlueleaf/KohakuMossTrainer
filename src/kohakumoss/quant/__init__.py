"""Quantized frozen base weights (int8 / fp8 weight-only) and fp8 / MXFP8 compute
(frozen bases and trainable caches) for MossLinear."""

from kohakumoss.quant.comfy import attach_quantized, split_quantized
from kohakumoss.quant.compute import ComputeCache, ComputeWeight
from kohakumoss.quant.convert import (
    QuantLinear,
    attach_compute,
    default_filter,
    quantize_model,
    quantize_torch_linears,
)
from kohakumoss.quant.weights import QuantWeight, dequantize, quantize

__all__ = [
    "ComputeCache",
    "ComputeWeight",
    "QuantLinear",
    "QuantWeight",
    "attach_compute",
    "quantize_torch_linears",
    "attach_quantized",
    "default_filter",
    "dequantize",
    "quantize",
    "quantize_model",
    "split_quantized",
]
