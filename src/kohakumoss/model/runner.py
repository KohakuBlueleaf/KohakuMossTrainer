"""Reference BlockRunner: calls the block directly."""

from torch import Tensor, nn

from kohakumoss.types.api import BlockCtx


class PlainRunner:
    """`run(i, block, x, ctx) = block(x, ctx)`; the other runners live in training.memory."""

    def run(self, index: int, block: nn.Module, x: Tensor, ctx: BlockCtx) -> Tensor:
        return block(x, ctx)
