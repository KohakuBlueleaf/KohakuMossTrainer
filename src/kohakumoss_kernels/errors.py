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


TOOLCHAIN_MODULES = ("triton", "tvm", "tilelang", "torch._inductor")
TOOLCHAIN_TEXT = (
    "compiler",
    "compilation",
    "nvcc",
    "ptxas",
    "nvrtc",
    "toolchain",
    "cl.exe",
    "msvc",
    "out of resources",
    "lowering",
    "codegen",
)


class BackendUnavailable(RuntimeError):
    """A backend cannot run `op` here; `errors` maps each attempt to its error text.
    `persist`: the cause is the build / toolchain here (recorded on disk for later
    processes), not one call's arguments. `problem`: the one problem (tuned kernel and
    key) that failed, None when the failure is not tied to a problem."""

    def __init__(
        self,
        op: str,
        errors: dict[str, str],
        what: str = "unavailable",
        persist: bool = True,
        problem: str | None = None,
    ):
        self.op = op
        self.errors = dict(errors)
        self.persist = persist
        self.problem = problem
        super().__init__(f"{op}: {what}: {self.errors}")


class ConfigsFailed(BackendUnavailable):
    """Every shortlisted config of a tuned kernel failed to compile or run on one
    problem (`problem`: "<kernel>:<key>")."""

    def __init__(
        self,
        kernel: str,
        errors: dict[str, str],
        persist: bool = True,
        problem: str | None = None,
    ):
        super().__init__(
            kernel, errors, "every shortlisted config failed", persist, problem
        )


class InputError(ValueError):
    """A backend failed on arguments the eager reference rejects too: a caller bug,
    never a backend failure (no fallback, nothing recorded)."""

    def __init__(self, op: str, backend: str, err: BaseException, eager: BaseException):
        super().__init__(
            f"{op}: backend {backend} failed on arguments the eager reference also "
            f"rejects: {describe(err)[:1000]}; eager: {describe(eager)[:1000]}"
        )


def is_build_error(err: BaseException) -> bool:
    """`err` says the code cannot build / launch here (out-of-memory never does)."""
    return isinstance(err, BUILD_ERRORS) and not isinstance(err, torch.OutOfMemoryError)


def is_toolchain_error(err: BaseException) -> bool:
    """`err` comes from building code here (compiler, toolchain, resource limits), so it
    holds for every later process with the same environment."""
    if isinstance(err, BackendUnavailable):
        return err.persist
    if isinstance(err, torch.OutOfMemoryError):
        return False
    if isinstance(err, OSError | UnicodeError):
        return True
    if TritonError is not None and isinstance(err, TritonError):
        return True
    if type(err).__module__.startswith(TOOLCHAIN_MODULES):
        return True
    if type(err).__name__ == "BackendCompilerFailed":
        return True
    text = str(err).lower()
    return any(t in text for t in TOOLCHAIN_TEXT)


STICKY_TEXT = (
    "illegal memory access",
    "illegal address",
    "misaligned address",
    "illegal instruction",
    "device-side assert",
    "unspecified launch failure",
    "hardware stack error",
    "cuda error: an illegal",
)


def is_sticky_error(err: BaseException, device=None) -> bool:
    """`err` left the device context unusable (a sticky CUDA error such as an illegal
    memory access): its text names one, or synchronising `device` raises. A fallback
    onto that context would fail too."""
    text = str(err).lower()
    if any(t in text for t in STICKY_TEXT):
        return True
    if device is None or torch.device(device).type != "cuda":
        return False
    try:
        torch.cuda.synchronize(device)
    except RuntimeError:
        return True
    return False


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
    return BackendUnavailable(
        op, {where or op: describe(err)}, persist=is_toolchain_error(err)
    )
