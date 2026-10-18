"""NONFINITE_PATIENCE: stop a run whose gradients stay non-finite (host values only)."""

from collections.abc import Callable

from kohakumoss.training.state import TrainState
from kohakumoss_common.calls import nothing


class NonFiniteGuard:
    """Raises FloatingPointError when non-finite gradients persist: `patience`
    consecutive non-finite steps with the loss scale at `floor` (a run without a dynamic
    scaler is always there), or the same non-finite parameters `patience` steps in a
    row. The message names the parameters (the step's non-finite report) and the file
    `record(dump_dir, step, names)` wrote the failing step to (StepRecorder.write);
    `forget()` runs after every finite step (StepRecorder.drop)."""

    every = 1

    def __init__(
        self,
        patience: int,
        floor: float,
        dump_dir: str = "",
        record: Callable = nothing,
        forget: Callable[[], None] = nothing,
    ):
        self.patience = patience
        self.floor = floor
        self.dump_dir = dump_dir
        self.record = record
        self.forget = forget
        self.run = 0
        self.same = 0
        self.last: tuple[str, ...] = ()

    def on_step_end(self, state: TrainState, out) -> None:
        if state.ok and state.nonfinite_params == 0:
            self.run, self.same, self.last = 0, 0, ()
            self.forget()
            return
        self.run += 1
        names = tuple(state.nonfinite_names)
        repeated = bool(names) and names == self.last
        self.same = self.same + 1 if repeated else int(bool(names))
        self.last = names
        at_floor = self.run >= self.patience and state.scale <= self.floor
        if at_floor or self.same >= self.patience:
            why = (
                f"{self.run} consecutive non-finite steps at loss scale {state.scale}"
                if at_floor
                else f"the same parameters non-finite {self.same} steps in a row"
            )
            path = self.record(self.dump_dir, state.step, names)
            saved = f"; the step's batch: {path}" if path is not None else ""
            raise FloatingPointError(
                f"step {state.step}: {why} ({state.nonfinite_params} parameters this "
                f"step; first: {', '.join(names) or 'loss only'}). Training would "
                f"continue with these parameters frozen. Reports: {self.dump_dir}"
                f"{saved}"
            )
