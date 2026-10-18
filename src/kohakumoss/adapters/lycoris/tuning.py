"""LyCORIS kernel tile tuning set from the run's TUNE, through LyCORIS's process switch
`lycoris.kernels.set_tuning`. See docs/internals/tuning.md#tune."""

from lycoris.kernels import set_tuning


def set_lycoris_tuning(on: bool) -> bool | None:
    """`on`: LyCORIS times a missing tile plan once and stores it; off: it binds its
    stored plan, else the planner's first, and times nothing. Returns the previous
    setting (pass it to `restore_lycoris_tuning`)."""
    return set_tuning(on)


def restore_lycoris_tuning(previous: bool | None) -> None:
    """Put back a setting `set_lycoris_tuning` returned (None: LyCORIS reads its
    environment switch again)."""
    set_tuning(previous)
