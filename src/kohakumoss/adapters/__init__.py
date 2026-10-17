"""Adapters over the lycoris-lora package: targeting, bound forwards, export, inference."""

from importlib.metadata import PackageNotFoundError, version

LYCORIS_MIN = "4.1.0.dev0"


def _lycoris_version() -> str:
    try:
        return version("lycoris-lora")
    except PackageNotFoundError:
        return "no installed distribution"


try:
    from lycoris.kernels import set_half_matmul as _set_half_matmul  # noqa: F401
    from lycoris.kernels import tier as _tier  # noqa: F401
except ImportError as e:
    raise ImportError(
        f"kohakumoss needs lycoris-lora >= {LYCORIS_MIN} (lycoris.kernels.tier, "
        f"set_half_matmul); found {_lycoris_version()}"
    ) from e

from kohakumoss.adapters.adapter_set import FORMATS, LycoAdapterSet  # noqa: E402
from kohakumoss.adapters.apply import apply  # noqa: E402
from kohakumoss.adapters.inference import (  # noqa: E402
    AdapterStack,
    describe,
    load,
    load_adapters,
)
from kohakumoss.adapters.kinds import Kind, register  # noqa: E402
from kohakumoss.adapters.presets import (  # noqa: E402
    BASES,
    MODIFIERS,
    resolve_preset,
)
from kohakumoss.adapters.spec import AdapterSpec, spec_from_knobs  # noqa: E402
from kohakumoss.adapters.tlora import set_token_mask, token_mask  # noqa: E402

__all__ = [
    "BASES",
    "FORMATS",
    "MODIFIERS",
    "AdapterSpec",
    "AdapterStack",
    "Kind",
    "LycoAdapterSet",
    "apply",
    "describe",
    "load",
    "load_adapters",
    "register",
    "resolve_preset",
    "set_token_mask",
    "spec_from_knobs",
    "token_mask",
]
