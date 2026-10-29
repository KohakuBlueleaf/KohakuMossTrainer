"""Reader-norm sites of the MOSS backbone and op-level taps on them.

A site is one RMSNorm input, in forward order: the text-refiner blocks `t{i}.1` /
`t{i}.2`, the joint blocks `j{i}.1` / `j{i}.2`, the head `h` (image tokens)."""

from collections.abc import Callable
from dataclasses import replace

from torch import Tensor

from kohakumoss.model.opset import OpSet


def site_names(text_blocks: int, depth: int) -> list[str]:
    """Every site in forward order."""
    names = [f"t{t}.{k}" for t in range(text_blocks) for k in (1, 2)]
    names += [f"j{i}.{k}" for i in range(depth) for k in (1, 2)]
    return names + ["h"]


def norm_path(site: str) -> str:
    """The norm module that reads `site`."""
    match site[0]:
        case "t":
            block, k = site[1:].split(".")
            return f"text_embedder.blocks.{block}.norm{k}"
        case "j":
            block, k = site[1:].split(".")
            return f"layers.{block}.norm{k}"
    return "final_norm_video"


def model_sites(model) -> list[str]:
    return site_names(len(model.text_embedder.blocks), model.cfg.depth)


class NormTaps:
    """`rms_norm` / `rms_norm_mod` ops that hand each norm input to `fn(site, x)` before
    calling the wrapped ops; sites are assigned by call order within one forward
    (`begin` resets, `end` checks that every site was seen once). `fn = None` taps
    nothing."""

    def __init__(self, sites: list[str], rms_norm: Callable, rms_norm_mod: Callable):
        self.sites = sites
        self.base_norm, self.base_mod = rms_norm, rms_norm_mod
        self.fn: Callable[[str, Tensor], None] | None = None
        self.pos = 0

    def begin(self) -> None:
        self.pos = 0

    def end(self) -> None:
        if self.fn is not None and self.pos != len(self.sites):
            raise RuntimeError(
                f"norm taps saw {self.pos} norm calls in one forward, expected "
                f"{len(self.sites)} (a fused norm or a block runner bypasses the ops)"
            )

    def _seen(self, x: Tensor) -> None:
        if self.fn is None:
            return
        if self.pos >= len(self.sites):
            raise RuntimeError(f"norm taps: more than {len(self.sites)} norm calls")
        self.fn(self.sites[self.pos], x.detach())
        self.pos += 1

    def rms_norm(self, x, weight, eps):
        self._seen(x)
        return self.base_norm(x, weight, eps)

    def rms_norm_mod(self, x, weight, shift, scale, seg, eps):
        self._seen(x)
        return self.base_mod(x, weight, shift, scale, seg, eps)

    def opset(self, ops: OpSet) -> OpSet:
        """`ops` with the norms routed through the taps and no fused linears."""
        return replace(
            ops, rms_norm=self.rms_norm, rms_norm_mod=self.rms_norm_mod, fused={}
        )


def tapped(model, ops: OpSet) -> NormTaps:
    """Bind `ops` with taps on every reader norm to `model`; returns the taps (forward
    pre/post hooks on the model call `begin` / `end`)."""
    taps = NormTaps(model_sites(model), ops.rms_norm, ops.rms_norm_mod)
    model.bind_ops(taps.opset(ops))
    model.register_forward_pre_hook(lambda mod, args: taps.begin())
    model.register_forward_hook(lambda mod, args, out: taps.end())
    return taps
