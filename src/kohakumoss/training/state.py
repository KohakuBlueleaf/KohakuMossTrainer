"""Mutable run state shared between the driver and its callbacks."""

from dataclasses import dataclass, field


@dataclass
class TrainState:
    """Host-side values only (no device tensors)."""

    step: int = 0
    tokens: int = 0
    samples: int = 0
    joint_sq: int = 0
    text_tokens: int = 0
    loss: float = float("nan")
    grad_norm: float = float("nan")
    ok: bool = True
    lr: float = 0.0
    scale: float = 1.0
    skipped: int = 0  # steps not applied
    nonfinite: int = 0  # steps with a non-finite gradient or loss
    nonfinite_params: int = 0  # parameters with a non-finite gradient this step
    nonfinite_names: list = field(default_factory=list)  # their first names this step
    batch_ids: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
