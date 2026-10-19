"""The host gradient sums of a trainable weight stream. See
docs/internals/offload.md#weight-streaming."""

from collections.abc import Callable

import torch
from torch import Tensor, nn

from kohakumoss.offload.optim.host_acc import HostAccumulator
from kohakumoss.offload.weights.streaming.device import DeviceStreams
from kohakumoss_common.calls import nothing


class GradSink:
    """One fp32 host gradient sum per trainable (block, key): a step's first gradient
    of a key replaces its sum, later ones add (`filled` marks the sums holding this
    step's gradients); on a device the D2H and the add run on a host worker thread
    (`HostAccumulator`), so backward does not wait. The bodies of `accumulate(key,
    grad)`, `drain()` and `land_block(index, keys)` are selected at construction."""

    def __init__(
        self,
        params: dict[tuple[int, str], nn.Parameter],
        blocks: int,
        dev: DeviceStreams,
        pinned: bool,
        stats: dict,
    ):
        device, dm, side = dev.device, dev.dm, dev.streamed
        self.params = params
        self.streamed = side
        self.pinned = pinned
        self.stats = stats
        self.host_grad: dict[tuple[int, str], Tensor] = {}
        self.filled: dict[tuple[int, str], bool] = {}
        self.ids = {key: n for n, key in enumerate(params)}
        self.block_keys: list[list[tuple[int, str]]] = [[] for _ in range(blocks)]
        for key in params:
            self.block_keys[key[0]].append(key)
        self.acc = None
        if not side:
            self._accumulate, self.drain, self._land_block = (
                GradSink._accumulate_host,
                nothing,
                nothing,
            )
            return
        self.acc = HostAccumulator(pinned, stats=stats)
        self.d2h = dm.Stream(device)
        self._accumulate, self.drain, self._land_block = (
            GradSink._accumulate_device,
            self.acc.drain,
            GradSink._land_block_device,
        )

    def accumulate(self, key: tuple[int, str], grad: Tensor) -> None:
        """Add `grad` to the host sum of `key` (replace it for the step's first)."""
        self._accumulate(self, key, grad)

    def land_block(self, index: int, keys) -> None:
        """Every queued gradient of one block's `keys` has landed in `host_grad`."""
        self._land_block(self, index, keys)

    def buffer(self, key: tuple[int, str]) -> Tensor:
        """The host sum of `key`, allocated on first use."""
        buf = self.host_grad.get(key)
        if buf is None:
            master = self.params[key]
            buf = torch.zeros(master.shape, dtype=torch.float32, pin_memory=self.pinned)
            self.host_grad[key] = buf
        return buf

    def _accumulate_host(self, key: tuple[int, str], grad: Tensor) -> None:
        buf = self.buffer(key)
        if self.filled.get(key, False):
            buf.add_(grad.detach())
        else:
            buf.copy_(grad.detach())
        self.filled[key] = True

    def _accumulate_device(self, key: tuple[int, str], grad: Tensor) -> None:
        """Queue `host_grad[key] += grad` (`=` for the step's first gradient; worker
        thread, backward does not wait)."""
        first = not self.filled.get(key, False)
        buf = self.buffer(key)
        self.acc.add(self.ids[key], grad, buf, self.d2h, first=first)
        self.filled[key] = True
        self.stats["bytes_d2h"] += grad.nbytes

    def _land_block_device(self, index: int, keys) -> None:
        """Every queued gradient of one block's `keys` has landed in `host_grad`."""
        for key in keys:
            self.acc.wait(self.ids[key])

    def expose(self, mine: Callable[[int], bool]) -> None:
        """This step's sums become the `.grad` of the masters of the blocks `mine`
        names."""
        for key, filled in self.filled.items():
            if filled and mine(key[0]):
                self.params[key].grad = self.host_grad[key]

    def zero(self) -> None:
        """End the step's sums: the next gradient of each key replaces its sum."""
        self.drain()
        for key in self.filled:
            self.filled[key] = False

    def close(self) -> None:
        """Stop the accumulator's worker thread (device streams only)."""
        if self.acc is not None:
            self.acc.close()
