"""Reference BlockRunner: calls the block directly."""

from torch import Tensor, nn

from kohakumoss.types.context import BlockCtx


class PlainRunner:
    """`run(i, block, x, ctx) = block(x, ctx)`; the other runners live in kohakumoss.memory.runners."""

    def bind(self, layers) -> "PlainRunner":
        return self

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        return block(x, ctx)
