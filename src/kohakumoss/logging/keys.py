"""Metric names: the training metrics a METRICS_LEVEL 'train' row keeps, and the grouped
name each metric has in wandb (`loss/`, `stream/`, `opt/`, `amp/`, `speed/`, `probe/`;
every other metric under `detail/`). `metrics.jsonl` keeps the plain names. See
docs/guides/logging.md#metric-names."""

GROUPED = {
    "loss": "loss/total",
    "loss_task": "loss/flow",
    "loss_distill": "loss/distill",
    "concentration": "loss/concentration",
    "rms_loss": "loss/rms",
    "peak_loss": "loss/peak",
    "stream_rms_q": "stream/rms",
    "stream_rms_q_block_max": "stream/rms_block_max",
    "stream_rms": "stream/rms_token",
    "stream_rms_block_max": "stream/rms_token_block_max",
    "stream_peak": "stream/peak",
    "lr": "opt/lr",
    "grad_norm": "opt/grad_norm",
    "loss_scale": "amp/scale",
    "skipped_steps": "amp/skipped",
    "step_time": "speed/step_time",
    "samples_per_s": "speed/img_per_s",
    "tokens_per_s": "speed/tokens_per_s",
}
PROBE_PREFIX = "grad_"


def is_training(name: str) -> bool:
    """Whether a 'train' row keeps metric `name`."""
    return name in GROUPED or name.startswith(PROBE_PREFIX)


def train_row(metrics: dict) -> dict:
    """The training metrics of a row (`is_training`)."""
    return {k: v for k, v in metrics.items() if is_training(k)}


def logged_name(name: str) -> str:
    """The grouped name of metric `name`: its group, `probe/<name>` for the gradient
    probe, else `detail/<name>`."""
    grouped = GROUPED.get(name)
    if grouped is not None:
        return grouped
    if name.startswith(PROBE_PREFIX):
        return f"probe/{name[len(PROBE_PREFIX):]}"
    return f"detail/{name}"


def named_row(metrics: dict) -> dict:
    """`metrics` under their grouped names (`logged_name`)."""
    return {logged_name(k): v for k, v in metrics.items()}
