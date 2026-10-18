"""The run's kernel prewarm actions, bound at build time under TUNE="table" ("off"
binds none, nothing is timed): the backbone prewarm before step 0 (`ModelPrewarm` over
the run's weightless twin), the adapter autotuned-op prewarm, and the online text
encoder's prewarm right before its first pass (`TextPrewarm`). The step callbacks live
in `training.callbacks.kernels`. See docs/internals/tuning.md#prewarm.
"""

import time

import torch

from kohakumoss.adapters.bind.autotuned import token_buckets
from kohakumoss.encoders.text.prewarm import prewarm_text
from kohakumoss.precision import autocast_factory
from kohakumoss.training.build.objective import build_loss, is_masked
from kohakumoss.training.build.shapes import run_shapes
from kohakumoss.training.callbacks.kernels import late_prewarm
from kohakumoss.tuning.lengths import planned_lengths, planned_segments
from kohakumoss.tuning.prewarm import (
    AdapterPrewarm,
    ModelPrewarm,
    log_prewarm,
    note_peak,
    reset_peaks,
    tunes,
)
from kohakumoss_common.calls import nothing
from kohakumoss_data import DataConfig
from kohakumoss_kernels.hardware.memory import release
from kohakumoss_kernels.tuning import decisions
from kohakumoss_kernels.tuning.window import tuning_window


def build_prewarm(k: dict, built, prec, device: torch.device, dataset=None):
    """The prewarm action of the run (a no-op unless TUNE times kernels): recorded
    from the weightless twin of `built.backbone` on `device`, the attention segment
    counts from `dataset`'s smallest planned grid and its shortest and longest caption
    estimates (None: the area target's grid, the caption-free template and
    MAX_TEXT_TOKENS); the step-1 report starts from this run's prewarm peaks only."""
    if not tunes(k):
        return nothing
    reset_peaks()
    cfg = DataConfig.from_knobs(k)
    sample, batch = planned_lengths(cfg)
    text_min = 0 if dataset is None else dataset.text_tokens_min()
    text_max = cfg.max_text_tokens if dataset is None else dataset.text_tokens_max()
    image_min = 0 if dataset is None else dataset.image_tokens_min()
    shortest, segments = planned_segments(cfg, text_min, image_min)
    warm = ModelPrewarm(
        run_shapes(k, prec, built.plan, device, built.backbone),
        build_loss(k, built),
        autocast_factory(prec, device),
        prec.compute,
        sample,
        batch,
        segments=segments,
        shortest=shortest,
        text_max=max(text_max, cfg.text_overhead_tokens),
        text_shortest=max(text_min, cfg.text_overhead_tokens),
        masked=is_masked(k),
    )
    return warm


class TextPrewarm:
    """The online text encoder's prewarm (`prewarm_text` on its weightless twin), run
    once right before its first pass, inside a tuning window (`window.tuning_window`):
    every pack bucket of its plan with rows up to `max_text_tokens` and the length
    bucket of the data's longest caption estimate (`data_tokens`, 0: unknown); logged
    as a prewarm table, its peak kept for the step-1 report, its workspaces returned to
    the driver after it."""

    def __init__(self, te, max_text_tokens: int, data_tokens: int = 0):
        self.te = te
        self.max_text_tokens = max_text_tokens
        self.data_tokens = data_tokens

    def __call__(self) -> None:
        device = self.te.device
        start = time.perf_counter()
        with tuning_window():
            text = prewarm_text(self.te, self.max_text_tokens, self.data_tokens)
        reports = {f"text {n}": r for n, r in text.items()}
        took = time.perf_counter() - start
        if reports:
            log_prewarm(reports, took, device)
            note_peak(device, reports)
            decisions.flush("after the text-encoder prewarm")
        release(device)
        late_prewarm(device, time.perf_counter() - start)


def attach_text_prewarm(k: dict, built, dataset=None) -> None:
    """Under TUNE="table", the online text encoder (TEXT_CACHE "online" / "off") on an
    accelerator tunes its ops right before its first pass; `dataset` gives the longest
    caption estimate."""
    te = built.text_encoder
    if not tunes(k) or te is None or k["TEXT_CACHE"] == "offline":
        return
    if te.device.type == "cpu":
        return
    longest = 0 if dataset is None else dataset.text_tokens_max()
    te.before_first(TextPrewarm(te, k["MAX_TEXT_TOKENS"], longest))


def adapter_prewarm_devices(k: dict, device: torch.device) -> list[torch.device]:
    """Devices the adapter prewarm settles: every stage device when threaded (it runs
    on scratch inputs, not on the placed layers), else the training device; none on
    CPU."""
    if device.type == "cpu":
        return []
    if k["PARALLEL"] == "threaded":
        return [torch.device(device.type, i) for i in k["DEVICES"]]
    return [device]


def build_adapter_prewarm(k: dict, built, dtype: torch.dtype):
    """The adapter autotuned-op prewarm (a no-op unless TUNE times kernels)."""
    if not tunes(k):
        return nothing
    _, batch = planned_lengths(DataConfig.from_knobs(k))
    return AdapterPrewarm(built.trainable, dtype, token_buckets(batch))
