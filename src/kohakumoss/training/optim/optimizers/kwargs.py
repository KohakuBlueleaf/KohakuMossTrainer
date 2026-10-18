"""Optimiser knob checks: OPTIMIZER names an algorithm + implementation,
OPTIMIZER_KWARGS against the chosen builder's signature, rounding against PRECISION,
the EMA spec."""

import inspect

from kohakumoss.training.optim.optimizers import factory
from kohakumoss.training.optim.optimizers.factory import INTERNAL, OPTIMIZERS

FROM_KNOBS = ("groups", "device", "lr", "betas", "eps")
FROM_PLAN = ("pinned",)
NAMED = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def builder_kwargs(name) -> tuple[str, ...] | None:
    """The OPTIMIZER_KWARGS names the registered builder `name` takes (LR / BETAS / EPS
    and the memory plan's own arguments excluded); None for anything not registered
    (a dotted path), whose signature is not checked."""
    if not isinstance(name, str) or name not in OPTIMIZERS:
        return None
    params = inspect.signature(OPTIMIZERS.get(name)).parameters.values()
    skip = FROM_KNOBS + FROM_PLAN
    return tuple(p.name for p in params if p.kind in NAMED and p.name not in skip)


def kwargs_problems(k: dict) -> list[str]:
    """OPTIMIZER_KWARGS names the OPTIMIZER's builder does not take."""
    given = dict(k.get("OPTIMIZER_KWARGS") or {})
    allowed = builder_kwargs(k["OPTIMIZER"])
    if allowed is None or k["OPTIMIZER"] in INTERNAL:
        return []
    bad = sorted(set(given) - set(allowed))
    if not bad:
        return []
    takes = ", ".join(allowed) if allowed else "none"
    return [
        f"OPTIMIZER_KWARGS {bad} not taken by OPTIMIZER={k['OPTIMIZER']!r} (it takes: "
        f"{takes}; LR / BETAS / EPS have their own knobs)"
    ]


def name_problems(k: dict) -> list[str]:
    """An internal placement builder given as OPTIMIZER; adamw8bit without
    bitsandbytes."""
    opt = k["OPTIMIZER"]
    if opt in INTERNAL:
        return [
            f"OPTIMIZER={opt!r} is the placement of OPTIMIZER='adamw' under OPTIM_STATE "
            f"({INTERNAL[opt]!r}): set OPTIMIZER='adamw' and OPTIM_STATE={INTERNAL[opt]!r}"
        ]
    if opt == "adamw8bit" and factory.bnb is None:
        return [
            "OPTIMIZER='adamw8bit' needs the bitsandbytes package (extra `quant`); "
            "OPTIMIZER='adamw8bit_kernel' is KohakuMoss's own 8-bit AdamW"
        ]
    return []


def optimizer_problems(k: dict) -> list[str]:
    """Every OPTIMIZER / OPTIMIZER_KWARGS / rounding / EMA problem."""
    p = name_problems(k) + kwargs_problems(k)
    rounding = (k.get("OPTIMIZER_KWARGS") or {}).get("rounding", "nearest")
    if rounding == "stochastic" and k["PRECISION"] != "bf16":
        p.append("stochastic rounding is bf16-only: use rounding='kahan' for fp16")
    ema = k["EMA"]
    if not (ema in (None, 0, 0.0) or isinstance(ema, (int, float, list, tuple, dict))):
        p.append(
            f"EMA={ema!r}: 0 | decay | [decays] | {{'kind': 'power', 'sigma_rel': [...]}}"
        )
    if isinstance(ema, (int, float)) and ema and not 0 < ema < 1:
        p.append(f"EMA decay must be in (0, 1), got {ema}")
    return p
