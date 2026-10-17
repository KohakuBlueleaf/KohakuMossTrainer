"""Reduction mode of the block-op parameter gradients, `KOHAKUMOSS_REDUCE`.

- df: double-float (fp32 pair, ~48-bit significand) products and sums.
- kahan: fp32 products, Kahan-compensated sums (cheaper, less precise).

Both store their partials as fp32 pairs. See docs/internals/fp16.md.
"""

import os

MODES = ("df", "kahan")
MODE = os.environ.get("KOHAKUMOSS_REDUCE", "df")
if MODE not in MODES:
    raise ValueError(f"KOHAKUMOSS_REDUCE={MODE!r} not in {MODES}")
DF = MODE == "df"
