"""The compile-cache key tag of this package: `torch.compiler.config.cache_key_tag`
carries a hash of the package version and of every source that registers a custom op
or its autograd, so a cached compile never outlives the op code it was built from. Set
once per process by `apply()`, on the first kernel build. See
docs/internals/kernel-dev.md#compile-cache-key."""

import hashlib
import importlib.metadata
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MARKERS = ("custom_op(", "register_autograd(", "torch.library")
SEP = "|"


def source_tag(version: str, sources: Iterable[tuple[str, str]]) -> str:
    """`kohakumoss-<hash>` of `version` and the (path, text) sources, path-ordered."""
    h = hashlib.sha1(version.encode("utf-8"))
    for path, text in sorted(sources):
        h.update(b"\0" + path.encode("utf-8") + b"\0" + text.encode("utf-8"))
    return f"kohakumoss-{h.hexdigest()[:12]}"


def op_sources(root: Path = ROOT) -> list[tuple[str, str]]:
    """(relative path, text) of every module under `root` that registers an op."""
    out = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in MARKERS):
            out.append((path.relative_to(root).as_posix(), text))
    return out


def package_version() -> str:
    try:
        return importlib.metadata.version("kohakumoss")
    except importlib.metadata.PackageNotFoundError:
        return "0"


def with_tag(current: str, tag: str) -> str:
    """`current` with `tag` appended (`|`-separated), unchanged when already present."""
    if tag in current.split(SEP):
        return current
    return f"{current}{SEP}{tag}" if current else tag


@cache
def tag() -> str:
    """This package's tag (the sources are read once per process)."""
    return source_tag(package_version(), op_sources())


def apply() -> str:
    """Append `tag()` to torch's compile-cache key tag (idempotent); returns the tag."""
    cfg = torch.compiler.config
    cfg.cache_key_tag = with_tag(cfg.cache_key_tag, tag())
    return tag()
