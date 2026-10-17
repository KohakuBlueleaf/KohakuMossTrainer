"""Pieces both drivers share: batch grouping, per-step bookkeeping, checkpoint parts."""

import contextlib
from collections.abc import Callable, Iterable, Iterator

from kohakumoss.training.state import TrainState
from kohakumoss.training.step import image_token_count, image_tokens_per_sample


def no_accumulate(last: bool):
    """Gradient-sync context for a micro-batch (DDP passes no_sync for non-last ones)."""
    return contextlib.nullcontext()


def grouped(loader: Iterable, n: int) -> Iterator[list]:
    """Endless lists of `n` host batches (re-iterates the loader when it ends)."""
    while True:
        group = []
        for hb in loader:
            group.append(hb)
            if len(group) == n:
                yield group
                group = []


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


def record(state: TrainState, bundle) -> None:
    state.loss = bundle.last_loss
    state.grad_norm = bundle.last_norm
    state.ok = bundle.last_ok
    state.lr = bundle.lr
    state.scale = bundle.scaler.scale
    state.skipped = bundle.skipped
    state.nonfinite = bundle.nonfinite


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
