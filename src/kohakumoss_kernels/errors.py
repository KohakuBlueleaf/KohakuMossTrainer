"""Errors a backend raises when it cannot run an op here, and the error classes that
count as "this backend / config cannot build or launch" (never out-of-memory). See
docs/internals/kernel-dev.md."""

import torch

try:
    from triton.errors import TritonError
except ImportError:
    TritonError = None

BUILD_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    UnicodeError,
    *((TritonError,) if TritonError is not None else ()),
)
LAUNCH_ERRORS: tuple[type[BaseException], ...] = tuple(
    e for e in BUILD_ERRORS if e is not RuntimeError
)


class BackendUnavailable(RuntimeError):
    """A backend cannot run `op` here; `errors` maps each attempt to its error text."""

    def __init__(self, op: str, errors: dict[str, str], what: str = "unavailable"):
        self.op = op
        self.errors = dict(errors)
        super().__init__(f"{op}: {what}: {self.errors}")


class ConfigsFailed(BackendUnavailable):
    """Every shortlisted config of a tuned kernel failed to compile or run."""

    def __init__(self, kernel: str, errors: dict[str, str]):
        super().__init__(kernel, errors, "every shortlisted config failed")


def is_build_error(err: BaseException) -> bool:
    """`err` says the code cannot build / launch here (out-of-memory never does)."""
    return isinstance(err, BUILD_ERRORS) and not isinstance(err, torch.OutOfMemoryError)


def describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"[:2000]


def unavailable(op: str, err: BaseException, where: str = "") -> BackendUnavailable:
    """`err` as `BackendUnavailable` for `op` when it is a build / launch failure;
    re-raises anything else (out-of-memory, bugs) unchanged. Use as
    `except Exception as e: raise unavailable(op, e) from e`."""
    if isinstance(err, BackendUnavailable):
        return err
    if not is_build_error(err):
        raise err
    return BackendUnavailable(op, {where or op: describe(err)})
