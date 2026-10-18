"""Output of one TrainStep call."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class StepOut:
    """`loss` is the fp32 loss already normalised by the global token count."""

    loss: Tensor
    n_tokens: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)

    def detached(self) -> "StepOut":
        """The same values with no autograd graph attached (what outlives a backward)."""
        return StepOut(
            _detach(self.loss),
            _detach(self.n_tokens),
            {k: _detach(v) for k, v in self.metrics.items()},
        )


def _detach(v):
    return v.detach() if isinstance(v, Tensor) else v
