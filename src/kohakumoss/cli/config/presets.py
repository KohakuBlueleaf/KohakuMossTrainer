"""The preset layers under configs/ and the choice of one layer per axis.

Train layers live in configs/moss/<group>/<group>_<name>.py, sample layers in
configs/sample/<group>/<group>_<name>.py. VRAM layers are named by their size in GiB
(`vram_24g`, full finetune: `vram_24g_full`); the catalogue is read from the files. A
card selects a size it holds within `tolerance` (a fraction of that size). See
docs/guides/configs.md.
"""

import re
from pathlib import Path

LAYER_GROUPS = {
    "moss": ("algo", "data", "vram", "precision", "parallel", "weights"),
    "sample": ("precision",),
}
VRAM_RE = re.compile(r"(\d+)g(_full)?")
FULL = "full"
AUTO = "auto"
VRAM_TOLERANCE = 0.05


def holds(vram: float, size: float, tolerance: float) -> bool:
    """`vram` GiB counts as `size` GiB when it is at least (1 - tolerance) x size."""
    return vram >= size * (1.0 - tolerance)


def group_dir(root: Path, domain: str, group: str) -> Path:
    return root / "configs" / domain / group


def names(root: Path, domain: str, group: str) -> list[str]:
    """Layer names of one group (file stem without the `<group>_` prefix)."""
    prefix = f"{group}_"
    return sorted(
        p.stem[len(prefix) :]
        for p in group_dir(root, domain, group).glob(f"{prefix}*.py")
    )


def layer(root: Path, domain: str, group: str, name: str) -> Path:
    """Path of one layer; SystemExit naming the available ones when it is missing."""
    path = group_dir(root, domain, group) / f"{group}_{name}.py"
    if not path.is_file():
        have = ", ".join(names(root, domain, group))
        raise SystemExit(f"no {group} preset {name!r}; available: {have}")
    return path


def catalogue(root: Path) -> dict[str, dict[str, list[str]]]:
    """{domain: {group: names}} over LAYER_GROUPS."""
    return {
        domain: {g: names(root, domain, g) for g in groups}
        for domain, groups in LAYER_GROUPS.items()
    }


def vram_sizes(root: Path, full: bool) -> list[int]:
    """GiB sizes of the VRAM layers for adapters (`full=False`) or full finetune."""
    out = []
    for name in names(root, "moss", "vram"):
        found = VRAM_RE.fullmatch(name)
        if found and bool(found.group(2)) == full:
            out.append(int(found.group(1)))
    return sorted(out)


def vram_layer(
    root: Path, vram: float | str, full: bool, tolerance: float = VRAM_TOLERANCE
) -> str:
    """Train VRAM layer name: the largest preset size `vram` GiB holds (within
    `tolerance`); above the largest, `auto` (the memory planner picks); "auto" passes
    through."""
    if vram == AUTO:
        return AUTO
    sizes = vram_sizes(root, full)
    kind = "full-finetune" if full else "adapter"
    if not sizes:
        raise SystemExit(f"no {kind} VRAM presets under configs/moss/vram/")
    fits = [s for s in sizes if holds(vram, s, tolerance)]
    if not fits:
        raise SystemExit(
            f"{vram:g} GiB is below the smallest {kind} VRAM preset ({sizes[0]} GiB, "
            f"tolerance {tolerance:g}); pass --vram or --vram-tolerance"
        )
    if vram > sizes[-1]:
        return AUTO
    size = max(fits)
    return f"{size}g_full" if full else f"{size}g"


def parallel_layer(gpus: int, full: bool) -> str | None:
    """Parallel layer name for `gpus` devices: None for one, fsdp / ddp otherwise."""
    if gpus <= 1:
        return None
    return "fsdp" if full else "ddp"


def sample_vram_layer(
    root: Path, vram: float, tolerance: float = VRAM_TOLERANCE
) -> str:
    """The adapter VRAM layer whose DiT placement sampling on a `vram` GiB card uses:
    the card's training layer, the smallest one below it, `auto` above the largest."""
    sizes = vram_sizes(root, False)
    if sizes and not any(holds(vram, s, tolerance) for s in sizes):
        return f"{sizes[0]}g"
    return vram_layer(root, vram, False, tolerance)
