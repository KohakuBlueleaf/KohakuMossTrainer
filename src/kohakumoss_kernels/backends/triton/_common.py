"""Shared by Triton ops: launching inside `triton_op` traces and dtype mapping."""

import torch
from torch.library import wrap_triton

from kohakumoss_kernels.hardware.current import on_device

PLAIN = (torch.Tensor, torch.nn.Parameter)


class Launcher:
    """A `JITFunction` launched directly on real tensors (on their device) and through
    `wrap_triton` while a `triton_op` is traced (fake / functional tensors)."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self._launch(grid, args, kwargs)

    def _launch(self, grid, args, kwargs):
        first = next(a for a in args if isinstance(a, torch.Tensor))
        if type(first) in PLAIN:
            with on_device(first.device):
                return self.fn[grid](*args, **kwargs)
        return wrap_triton(self.fn)[grid](*args, **kwargs)


def eager_function(name: str, impl, setup, backward, nondiff: tuple[int, ...] = ()):
    """An autograd.Function over a triton_op's forward body `impl`, its `setup_context`
    and a backward that calls the backward body: the same computation without the
    custom-op dispatch, for calls outside torch.compile. `nondiff` indexes the outputs
    that get no gradient. See docs/internals/kernel-dev.md."""

    def forward(ctx, *inputs):
        out = impl(*inputs)
        setup(ctx, inputs, out)
        if nondiff:
            ctx.mark_non_differentiable(*(out[i] for i in nondiff))
            ctx.set_materialize_grads(False)
        return out

    return type(
        name,
        (torch.autograd.Function,),
        {"forward": staticmethod(forward), "backward": staticmethod(backward)},
    )
