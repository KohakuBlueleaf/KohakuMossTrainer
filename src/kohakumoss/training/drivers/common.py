"""Pieces both drivers share: batch grouping, per-step bookkeeping, checkpoint parts."""

import contextlib
from collections.abc import Callable, Iterable, Iterator

from kohakumoss.training.state import TrainState
from kohakumoss.training.step import image_token_count, image_tokens_per_sample
from kohakumoss_common.calls import nothing
from kohakumoss_kernels.registry import settle_guards


def no_accumulate(last: bool):
    """Gradient-sync context for a micro-batch (DDP passes no_sync for non-last ones)."""
    return contextlib.nullcontext()


def grouped(loader: Iterable, n: int) -> Iterator[list]:
    """Endless lists of `n` host batches; a group continues across loader passes, and a
    pass that yields no batch raises."""
    group = []
    while True:
        seen = False
        for hb in loader:
            seen = True
            group.append(hb)
            if len(group) == n:
                yield group
                group = []
        if not seen:
            raise RuntimeError("the training loader yielded no batch in a full pass")


def last_flags(n: int) -> tuple[bool, ...]:
    """Per micro-batch "is the last of the step" flags for GRAD_ACC = n."""
    return (False,) * (n - 1) + (True,)


def loss_count(group: list, count_fn: Callable = image_token_count) -> int:
    """A group's loss-token count (`count_fn` per host batch)."""
    return sum(count_fn(hb) for hb in group)


def account(state: TrainState, group: list) -> None:
    """Add a group's sample / token counts to `state`, after its batches were moved to
    the device (online text encoding fills `text_len` there)."""
    ids = []
    for hb in group:
        img = image_tokens_per_sample(hb)
        txt = hb.text_len.tolist()
        state.samples += len(img)
        state.text_tokens += sum(txt)
        state.joint_sq += sum((a + b) ** 2 for a, b in zip(img, txt))
        state.tokens += sum(img)
        ids += hb.ids.tolist()
    state.batch_ids = ids


def micro_steps(
    step, batcher, accumulate, lasts, bundle, group, denom, index: int, ema=None
):
    """Forward + backward of one optimiser step's micro-batches; the last StepOut, with
    no autograd graph attached (each micro-batch's graph ends with its backward).
    `ema.join` (an EMA updating in the background) runs before each backward."""
    join = getattr(ema, "join", nothing)
    out = None
    for hb, last in zip(group, lasts):
        batch = batcher(hb)
        with accumulate(last):
            out = step(batch, denom, index)
            join()
            bundle.backward(out.loss)
            out = out.detached()
    return out


def finish_step(state: TrainState, group: list, bundle, ema, runner, out) -> None:
    """After a group's backward: count it, apply the optimiser step (EMA on success),
    bind the plain call path of the kernels proven in this step (`settle_guards`),
    advance `state.step`, record the step and run the callbacks."""
    account(state, group)
    if bundle.step():
        ema.update(state.step)
    settle_guards()
    state.step += 1
    record(state, bundle)
    runner.run(state, out)


def record(state: TrainState, bundle) -> None:
    """Copy the bundle's host-side step results into `state`."""
    state.loss = bundle.last_loss
    state.grad_norm = bundle.last_norm
    state.ok = bundle.last_ok
    state.lr = bundle.lr
    state.scale = bundle.scaler.scale
    state.skipped = bundle.skipped
    state.nonfinite = bundle.nonfinite
    state.nonfinite_params = bundle.last_nonfinite_params
    state.nonfinite_names = [g["param"] for g in bundle.nonfinite_grads]


class Parts:
    """Collects what a checkpoint stores, from the live objects."""

    def __init__(
        self, trainable, ema, bundle, step, loader, extra: Callable[[], dict] = dict
    ):
        self.trainable = trainable
        self.ema = ema
        self.bundle = bundle
        self.step = step
        self.loader = loader
        self.extra = extra

    def __call__(self, state: TrainState) -> dict:
        return {
            "trainable": self.trainable.state_dict(),
            "trainable_meta": self.trainable.state_metadata(),
            "ema": self.ema.state_dict(),
            "ema_gammas": dict(
                zip(getattr(self.ema, "tags", ()), getattr(self.ema, "gammas", ()))
            ),
            "optim": self.bundle.state_dict(),
            "step_rng": self.step.rng_state(),
            "state": {
                "loader": self.loader.state_dict(),
                "tokens": state.tokens,
                "samples": state.samples,
                "joint_sq": state.joint_sq,
                "text_tokens": state.text_tokens,
                **self.extra(),
            },
        }

    def restore(self, ckpt: dict, state: TrainState) -> None:
        self.trainable.load_state_dict(ckpt["trainable"])
        self.ema.load_state_dict(ckpt["ema"])
        self.bundle.load_state_dict(ckpt["optim"])
        self.step.load_rng_state(ckpt["step_rng"])
        st = ckpt["state"]
        self.loader.load_state_dict(st["loader"])
        state.step = int(st["step"])
        for key in ("tokens", "samples", "joint_sq", "text_tokens"):
            setattr(state, key, int(st.get(key, 0)))

    def before_export(self) -> None:
        """Make the live trainable set the run's full trained state (every rank calls
        it; one process holds it already)."""
        return None


class ShardedOptimParts(Parts):
    """Parts whose optimiser state is sharded across ranks (DDP block owners of the
    streamed masters): every rank writes and reads its own `optim.<rank>.pt`."""

    def __call__(self, state: TrainState) -> dict:
        parts = super().__call__(state)
        parts["optim_rank"] = parts["optim"]
        parts["optim"] = {}
        return parts

    def restore(self, ckpt: dict, state: TrainState) -> None:
        super().restore({**ckpt, "optim": ckpt["optim_rank"]}, state)
