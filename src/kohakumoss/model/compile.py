"""COMPILE granularity: "off" (eager), "block" (each joint block's forward compiled
once per class, so every block of the model runs the same graph), "model" (the whole
backbone forward compiled, the block forwards inlined); the per-step glue around the
model (`compiled_glue`) compiles under both. See docs/concepts/training.md#compile."""

import functools
from collections.abc import Callable

import torch
from torch import nn

from kohakumoss_kernels import compile_setup

MODES = ("off", "block", "model")
_STATE = {"mode": "off"}


def set_mode(mode: str) -> None:
    """The COMPILE mode the decorated forwards follow from their next first call."""
    if mode not in MODES:
        raise ValueError(f"COMPILE={mode!r} not in {MODES}")
    _STATE["mode"] = mode


def compiled_block(func: Callable) -> Callable:
    """`func` (a block class's forward) compiled once, lazily, under COMPILE "block";
    called inside an enclosing compiled region it runs inline (traced into that
    graph); eager otherwise."""
    run = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal run
        if torch.compiler.is_compiling():
            return func(*args, **kwargs)
        if run is None:
            if _STATE["mode"] == "block":
                compile_setup.apply()
                run = torch.compile(func, dynamic=True)
            else:
                run = func
        return run(*args, **kwargs)

    return wrapper


def compiled_glue(func: Callable) -> Callable:
    """`func` (per-step math around the model: losses, penalties) compiled once,
    lazily, under COMPILE "block" or "model"; eager under "off"."""
    run = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal run
        if run is None:
            if _STATE["mode"] != "off":
                compile_setup.apply()
                run = torch.compile(func, dynamic=True)
            else:
                run = func
        return run(*args, **kwargs)

    return wrapper


def compile_model(backbone: nn.Module, mode: str) -> None:
    """Apply COMPILE `mode` to `backbone`: "model" compiles its forward in place, the
    other modes are carried by the decorated block forwards."""
    set_mode(mode)
    if mode == "model":
        compile_setup.apply()
        backbone.compile(dynamic=True)
