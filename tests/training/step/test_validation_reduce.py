"""Validation reports the global per-token mean (the ranks' shares reduced like the
training loss) on conditional samples only."""

import pytest
import torch

from kohakumoss.training.build.loop import validation_data_config
from kohakumoss.training.callbacks.evaluate import Validation
from kohakumoss.types.step import StepOut
from tests._util.negative import must_fail
from tests.training.knobs.test_knobs import runnable

pytestmark = pytest.mark.cpu


class _Step:
    """A rank's share of one batch's loss: 0.25 per batch at every sigma."""

    def __init__(self, sampler):
        self.noise_gen = torch.Generator()

    def denominator(self, batch):
        return torch.tensor(1.0)

    def __call__(self, batch, denom):
        return StepOut(torch.tensor(0.25), denom)


class _Modes:
    """What Validation reads of the backbone: `training`, `train()`, `eval()`."""

    training = True

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def eval(self) -> None:
        self.training = False


def _evaluate(reduce=None) -> dict:
    kw = {} if reduce is None else {"reduce_loss": reduce}
    val = Validation(1, _Step, [0, 1], (0.5, 0.75), 0, _Modes(), (None, None), **kw)
    return val.evaluate()


def test_ranks_shares_are_reduced():
    two_ranks = _evaluate(lambda t: t * 2)  # sum over 2 ranks holding equal shares
    assert two_ranks["val_loss"] == pytest.approx(0.5)
    assert two_ranks["val_loss@0.5"] == pytest.approx(0.5)
    assert _evaluate()["val_loss"] == pytest.approx(0.25)
    must_fail(
        lambda: _close(_evaluate()["val_loss"], two_ranks["val_loss"]),
        "rank 0's local share reported",
    )


def _close(a, b):
    assert a == pytest.approx(b)


def test_validation_data_has_no_text_dropout():
    k = runnable(TEXT_DROP=0.3, VALIDATION={"datasets": [{"source": "synthetic"}]})
    cfg = validation_data_config(k)
    assert cfg.text_drop == 0.0 and cfg.datasets == [{"source": "synthetic"}]
