"""The BaseWeights a WeightStream installs on its MossLinears (frozen, trainable,
int8 / fp8, fp8-compute) and the autograd functions that read the ring slots. See
docs/internals/offload.md (weight streaming)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

from kohakumoss.quant.compute import PARTS, ComputeWeight, compute_linear
from kohakumoss.quant.weights import dequantize
from kohakumoss_kernels import registry
from kohakumoss_kernels.fp8 import dequantize as fp8_dequantize
from kohakumoss_kernels.fp8.op import dgrad as fp8_dgrad


class _StreamedLinear(Function):
    """`F.linear(x, w, bias)` over a ring slot that saves only `x`: the backward reads
    the weight from the slot again (the block is re-streamed), so no slot is held in
    the autograd graph and a block needs no checkpointing to stream. Under autocast the
    slot is read in the autocast dtype (no full-precision weight copy)."""

    @staticmethod
    def forward(ctx, x, bias, handle):
        ctx.handle = handle
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x)
        kind = x.device.type
        cast = torch.is_autocast_enabled(kind)
        dtype = torch.get_autocast_dtype(kind) if cast else x.dtype
        return F.linear(x, handle.weight_now(dtype), bias)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        handle = ctx.handle
        dx = db = None
        if ctx.needs_input_grad[0]:
            dx = (g @ handle.weight_now(g.dtype)).to(x.dtype)
        if ctx.has_bias and ctx.needs_input_grad[1]:
            db = g.reshape(-1, g.shape[-1]).sum(0)
        handle.weight_grad(g, x)
        return dx, db, None


class _StreamedFp8Linear(Function):
    """`fp8_linear` over ring slots that saves nothing: the base is frozen, so the
    input gradient needs only the weight parts, read from the slot again in backward
    (the block is re-streamed); no slot is held in the autograd graph."""

    @staticmethod
    def forward(ctx, x, bias, handle):
        ctx.handle = handle
        ctx.x_dtype = x.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return compute_linear(handle.infer, handle, x, bias)

    @staticmethod
    def backward(ctx, g):
        h = ctx.handle
        g2 = g.reshape(-1, g.shape[-1]).contiguous()
        dx = db = None
        if ctx.needs_input_grad[0]:
            parts = (h.q, h.scale, h.q_t, h.scale_t, h.aux_t)
            dx = fp8_dgrad(g2, *parts, h.recipe, h.train.backend)
            dx = dx.reshape(*g.shape[:-1], dx.shape[-1]).to(ctx.x_dtype)
        if ctx.bias_dtype is not None and ctx.needs_input_grad[1]:
            db = g2.float().sum(0).to(ctx.bias_dtype)
        return dx, db, None


class StreamedWeight:
    """Frozen BaseWeight backed by the engine's ring slot for (block, key)."""

    def __init__(self, engine, block: int, key: str, shape):
        self.engine = engine
        self.block = block
        self.key = key
        self.shape = torch.Size(shape)

    def weight_now(self, dtype: torch.dtype) -> Tensor:
        """The slot's weight in `dtype`, outside autograd."""
        return self.engine.slot(self.block, self.key).to(dtype)

    def weight(self, dtype: torch.dtype) -> Tensor:
        return self.weight_now(dtype)

    def weight_grad(self, g: Tensor, x: Tensor) -> None:
        return None

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        return _StreamedLinear.apply(x, bias, self)

    def materialize(self) -> Tensor:
        return self.engine.host[self.block][self.key]


class StreamedParamWeight(StreamedWeight):
    """Trainable streamed weight: gradients flow to the host fp32 master."""

    def __init__(self, engine, block, key, shape, master: nn.Parameter):
        super().__init__(engine, block, key, shape)
        self.master = master

    def weight(self, dtype: torch.dtype) -> Tensor:
        w = self.engine.slot(self.block, self.key)
        return _SlotParam.apply(self.master, w, self.engine, (self.block, self.key)).to(
            dtype
        )

    def weight_grad(self, g: Tensor, x: Tensor) -> None:
        """g^T x in the slot's dtype, handed to the engine like the slot gradient."""
        slot = self.engine.slot(self.block, self.key)
        g2 = g.reshape(-1, g.shape[-1]).to(slot.dtype)
        x2 = x.reshape(-1, x.shape[-1]).to(slot.dtype)
        self.engine.on_grad((self.block, self.key), g2.t() @ x2)

    def materialize(self) -> Tensor:
        return self.master.detach()


class StreamedQuantWeight(StreamedWeight):
    """Streamed int8/fp8 base: the ring carries `q` and `scale`, dequantized on use."""

    def weight_now(self, dtype: torch.dtype) -> Tensor:
        q = self.engine.slot(self.block, self.key + ":q")
        scale = self.engine.slot(self.block, self.key + ":scale")
        return dequantize(q, scale, dtype)

    def materialize(self) -> Tensor:
        host = self.engine.host[self.block]
        return dequantize(
            host[self.key + ":q"], host[self.key + ":scale"], torch.float32
        )


class StreamedComputeWeight(StreamedWeight):
    """Streamed fp8-compute base: the ring carries every stored part; `linear` runs
    the gradient-free `fp8_linear_infer` forward and re-reads the slot parts for the
    input gradient (`fp8_linear`'s backend launcher), bound to the ring's device and
    `dtype`."""

    def __init__(self, engine, block, key, base: ComputeWeight, dtype):
        super().__init__(engine, block, key, base.shape)
        self.recipe = base.recipe
        self.present = frozenset(base.tensors())
        device, recipe = engine.device, self.recipe
        self.infer = registry.resolve("fp8_linear_infer", device, dtype, recipe=recipe)
        self.train = registry.resolve("fp8_linear", device, dtype, recipe=recipe)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        return _StreamedFp8Linear.apply(x, bias, self)

    def __getattr__(self, name: str):
        if name not in PARTS:
            raise AttributeError(name)
        if name not in self.present:
            return None
        return self.engine.slot(self.block, f"{self.key}:{name}")

    def weight(self, dtype: torch.dtype) -> Tensor:
        return fp8_dequantize(self.q, self.scale, self.recipe, dtype)

    def materialize(self) -> Tensor:
        host = self.engine.host[self.block]
        q, scale = host[f"{self.key}:q"], host[f"{self.key}:scale"]
        return fp8_dequantize(q, scale, self.recipe, torch.float32)


class _Enter(Function):
    @staticmethod
    def forward(ctx, x, engine, index):
        ctx.engine, ctx.index = engine, index
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.engine.release(ctx.index)
        return grad, None, None


class _Exit(Function):
    @staticmethod
    def forward(ctx, y, engine, index):
        ctx.engine, ctx.index = engine, index
        return y.view_as(y)

    @staticmethod
    def backward(ctx, grad):
        engine = ctx.engine
        engine.ensure(ctx.index)
        engine.prefetch(ctx.index - 1)
        engine.on_block_backward(ctx.index)
        return grad, None, None


class _SlotParam(Function):
    @staticmethod
    def forward(ctx, param, slot, engine, key):
        ctx.engine, ctx.key = engine, key
        return slot.view_as(slot)

    @staticmethod
    def backward(ctx, grad):
        ctx.engine.on_grad(ctx.key, grad)
        return None, None, None, None
