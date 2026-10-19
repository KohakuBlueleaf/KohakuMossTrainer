"""Single-process multi-GPU pipeline (no process group): layer ranges on the devices
of one process. See docs/guides/multi-gpu.md (threaded pipeline)."""

import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import fields, replace

import torch
from torch import Tensor, nn

from kohakumoss.memory.runners import PlainRunner
from kohakumoss.types.context import BlockCtx
from kohakumoss.types.seqinfo import SeqInfo
from kohakumoss_common.calls import nothing


def ctx_to(ctx: BlockCtx, device: torch.device) -> BlockCtx:
    """The block context (or a subclass) with every tensor on `device`."""
    moved = {}
    for f in fields(ctx):
        value = getattr(ctx, f.name)
        match value:
            case Tensor():
                moved[f.name] = value.to(device, non_blocking=True)
            case SeqInfo():
                moved[f.name] = value.to(device)
            case _:
                moved[f.name] = value
    return replace(ctx, **moved)


def even_bounds(n: int, k: int) -> list[tuple[int, int]]:
    edges = [round(i * n / k) for i in range(k + 1)]
    return list(zip(edges[:-1], edges[1:]))


class ThreadedPipeline:
    """Runs `model.prologue -> layers -> model.epilogue` across `devices`.

    `opset_for(device)` binds each stage's blocks to that device's kernels;
    `schedule` is `gpipe` (all forwards, then each micro-batch's backward in reverse
    order) or `async`. `install(state)` puts a micro-batch's adapter step state back
    before its forward and its gpipe backward.
    """

    def __init__(
        self,
        model: nn.Module,
        devices: Sequence[torch.device | str],
        bounds: list[tuple[int, int]] | None = None,
        epilogue_modules: Sequence[nn.Module] = (),
        runner=None,
        opset_for: Callable | None = None,
        schedule: str = "gpipe",
    ):
        self.model = model
        self.install: Callable = nothing
        self.devices = [torch.device(d) for d in devices]
        self.bounds = bounds or even_bounds(len(model.layers), len(self.devices))
        if len(self.bounds) != len(self.devices):
            raise ValueError("one layer range per device")
        self.runner = (runner or PlainRunner()).bind(model.layers)
        layer_ids = {id(p) for p in model.layers.parameters()}
        tail_ids = {id(p) for m in epilogue_modules for p in m.parameters()}
        for p in model.parameters():
            if id(p) not in layer_ids and id(p) not in tail_ids:
                p.data = p.data.to(self.devices[0])
        for b in model.buffers():
            b.data = b.data.to(self.devices[0])
        for m in epilogue_modules:
            m.to(self.devices[-1])
        stages = []
        for dev, (lo, hi) in zip(self.devices, self.bounds):
            ops = opset_for(dev) if opset_for is not None else None
            for i in range(lo, hi):
                model.layers[i].to(dev)
                if ops is not None:
                    model.layers[i].bind(ops)
            stages.append((dev, tuple((i, model.layers[i]) for i in range(lo, hi))))
        self._head = stages[0][1]
        self._rest = tuple(stages[1:])
        match schedule:
            case "gpipe":
                self._step = ThreadedPipeline._gpipe
            case "async":
                self._step = ThreadedPipeline._async
            case _:
                raise ValueError(f"threaded schedule {schedule!r} not in gpipe|async")

    def forward(self, *inputs) -> Tensor:
        x, ctx = self.model.prologue(*inputs)
        run = self.runner.run
        for i, block in self._head:
            x = run(i, block, x, ctx)
        for dev, blocks in self._rest:
            x = x.to(dev, non_blocking=True)
            ctx = ctx_to(ctx, dev)
            for i, block in blocks:
                x = run(i, block, x, ctx)
        return self.model.epilogue(x, ctx)

    def step(
        self,
        microbatches: Sequence[tuple[tuple, object]],
        loss_fn: Callable[[Tensor, object], Tensor],
        states: Sequence | None = None,
    ) -> Tensor:
        """Forward + backward over `(inputs, extra)` micro-batches; returns the summed
        loss (detached). `loss_fn(pred, extra)` already normalises by the step's tokens;
        `states[k]` is micro-batch k's adapter step state (None: nothing to install).
        """
        states = list(states) if states is not None else [None] * len(microbatches)
        return self._step(self, microbatches, loss_fn, states)

    def _gpipe(self, microbatches, loss_fn, states) -> Tensor:
        losses = []
        for (inp, ex), state in zip(microbatches, states):
            self.install(state)
            losses.append(loss_fn(self.forward(*inp), ex))
        for loss, state in zip(reversed(losses), reversed(states)):
            self.install(state)
            loss.float().backward()
        return torch.stack([loss.detach().float() for loss in losses]).sum()

    def _async(self, microbatches, loss_fn, states) -> Tensor:
        jobs: queue.Queue = queue.Queue(maxsize=2)
        errors: list[BaseException] = []

        def worker():
            while True:
                loss = jobs.get()
                if loss is None:
                    return
                try:
                    loss.backward()
                except BaseException as err:
                    errors.append(err)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        losses = []
        try:
            for (inp, ex), state in zip(microbatches, states):
                self.install(state)
                loss = loss_fn(self.forward(*inp), ex)
                losses.append(loss.detach().float())
                jobs.put(loss)
        finally:
            jobs.put(None)
            thread.join()
        if errors:
            raise errors[0]
        return torch.stack([v.to(self.devices[-1]) for v in losses]).sum()
