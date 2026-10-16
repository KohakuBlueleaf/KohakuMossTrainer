"""Output of one TrainStep call."""

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class StepOut:
    """`loss` is the fp32 loss already normalised by the global token count."""

    loss: Tensor
    n_tokens: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)
