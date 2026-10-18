"""`grad_offload_wait_s` in the transfer metric rows, model-free (counter-holding
stand-ins for the engines): the staging-ring wait per step since the last log point,
present only when a source carries the counter."""

from types import SimpleNamespace

import pytest

from kohakumoss.offload.optim.host_acc import RING_WAIT
from kohakumoss.training.callbacks.transfer import TransferStats
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu


def _source(wait: bool) -> SimpleNamespace:
    stats = {"bytes_h2d": 0, "bytes_d2h": 0}
    if wait:
        stats[RING_WAIT] = 0.0
    return SimpleNamespace(stats=stats)


def assert_wait_rows(make) -> None:
    grads, stream, plain = _source(True), _source(True), _source(False)
    cb = make(1, [grads, stream, plain])
    grads.stats[RING_WAIT] += 0.3
    stream.stats[RING_WAIT] += 0.1
    state = SimpleNamespace(step=2, metrics={})
    cb.on_step_end(state, None)
    assert state.metrics["grad_offload_wait_s"] == pytest.approx(0.2)
    grads.stats[RING_WAIT] += 0.5
    state = SimpleNamespace(step=3, metrics={})
    cb.on_step_end(state, None)
    assert state.metrics["grad_offload_wait_s"] == pytest.approx(0.5)
    none = make(1, [plain])
    state = SimpleNamespace(step=1, metrics={})
    none.on_step_end(state, None)
    assert "grad_offload_wait_s" not in state.metrics


class _Cumulative(TransferStats):
    """Reports the total wait, not the wait since the last log point."""

    def on_step_end(self, state, out) -> None:
        super().on_step_end(state, out)
        if self.waited is not None:
            state.metrics["grad_offload_wait_s"] = self.waited()


def test_wait_metric_per_step():
    assert_wait_rows(TransferStats)
    must_fail(lambda: assert_wait_rows(_Cumulative), "cumulative wait")
