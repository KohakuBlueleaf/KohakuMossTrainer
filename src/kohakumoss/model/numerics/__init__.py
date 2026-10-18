"""Numerics: parameter dtypes, residual-stream fold, fp16 fixes, saturation counters."""

from kohakumoss.model.numerics.fold import (
    adopt_fold,
    fold,
    fold_record,
    output_row_scales,
    record_row_scales,
    same_fold,
    unfold,
)
from kohakumoss.model.numerics.fp16fix import (
    MODES as FP16_FIX_MODES,
)
from kohakumoss.model.numerics.fp16fix import (
    apply_fp16_fix,
    carry_fold,
    choose_factors,
    clamp_counts,
    clear_fp16_fix,
    fix_problems,
)
from kohakumoss.model.numerics.keep_fp32 import cast_params
from kohakumoss.model.numerics.saturation import SaturationCounters

__all__ = [
    "FP16_FIX_MODES",
    "SaturationCounters",
    "adopt_fold",
    "carry_fold",
    "cast_params",
    "fold_record",
    "record_row_scales",
    "same_fold",
    "apply_fp16_fix",
    "choose_factors",
    "clamp_counts",
    "clear_fp16_fix",
    "fix_problems",
    "fold",
    "output_row_scales",
    "unfold",
]
