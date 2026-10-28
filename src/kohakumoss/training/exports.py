"""The run's export files under SAVE_FORMAT (`comfy` | `hf` | `both` | `none`),
SAVE_MERGED and SAVE_FUSED_QKV, written at the end of a run and, with
CHECKPOINT_EXPORT, next to every checkpoint; the EXPORT_FORMATS spelling maps onto
them. See docs/guides/export-to-comfy.md."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from kohakumoss.logging.log import get_logger, hl
from kohakumoss.memory.estimate import total_params
from kohakumoss.model.io.load import stores_fused_qkv
from kohakumoss.model.numerics import output_row_scales
from kohakumoss.training.weights import component
from kohakumoss.weights import DIT
from kohakumoss_common.units import GIB

SAVE_FORMATS = ("comfy", "hf", "both", "none")
KINDS = {"comfy": ("comfy",), "hf": ("hf",), "both": ("comfy", "hf"), "none": ()}
FUSED_CHOICES = ("auto", True, False)
ADAPTER_OF = {"comfy": "comfy", "lycoris": "hf"}
MERGED_OF = {"merged": "comfy", "diffusers": "hf"}
EXPORT_FORMATS = (*ADAPTER_OF, *MERGED_OF)
SAVE_BYTES = 2
log = get_logger(__name__)


@dataclass(frozen=True)
class ExportFile:
    """One export: file name suffix, the trainable's export format, whether it is a
    folder, and whether q / k / v are written fused."""

    suffix: str
    fmt: str
    folder: bool = False
    fused: bool = False

    def name(self, run: str) -> str:
        return f"{run}-{self.suffix}" + ("" if self.folder else ".safetensors")


def from_export_formats(formats, full: bool) -> tuple[str, bool]:
    """(SAVE_FORMAT, SAVE_MERGED) of an EXPORT_FORMATS list: `comfy` / `merged` are the
    ComfyUI files, `lycoris` / `diffusers` the HF ones; a merged format on an adapter
    run sets SAVE_MERGED; an empty list is `none`."""
    adapter = {ADAPTER_OF[f] for f in formats if f in ADAPTER_OF}
    merged = {MERGED_OF[f] for f in formats if f in MERGED_OF}
    kinds = adapter | merged
    if not kinds:
        return "none", False
    save = "both" if kinds == {"comfy", "hf"} else kinds.pop()
    return save, bool(merged) and not full


def export_problems(k: dict) -> list[str]:
    """SAVE_FORMAT, SAVE_MERGED, SAVE_FUSED_QKV, CHECKPOINT_EXPORT and EXPORT_FORMATS
    values."""
    p = []
    if k["SAVE_FORMAT"] not in SAVE_FORMATS:
        p.append(f"SAVE_FORMAT={k['SAVE_FORMAT']!r} not in {SAVE_FORMATS}")
    if k["SAVE_FUSED_QKV"] not in FUSED_CHOICES:
        p.append(f"SAVE_FUSED_QKV={k['SAVE_FUSED_QKV']!r} not in {FUSED_CHOICES}")
    for name in ("SAVE_MERGED", "CHECKPOINT_EXPORT"):
        if not isinstance(k[name], bool):
            p.append(f"{name} must be True or False, got {k[name]!r}")
    formats = k.get("EXPORT_FORMATS")
    if formats is not None:
        bad = [f for f in formats if f not in EXPORT_FORMATS]
        if bad:
            p.append(
                f"EXPORT_FORMATS has unknown formats {bad}; valid {EXPORT_FORMATS}"
            )
    return p


def with_export_formats(k: dict) -> dict:
    """The knobs with a set EXPORT_FORMATS mapped onto SAVE_FORMAT / SAVE_MERGED (one
    INFO line naming the mapping); unchanged when it is None."""
    formats = k.get("EXPORT_FORMATS")
    if formats is None or [f for f in formats if f not in EXPORT_FORMATS]:
        return k
    save, merged = from_export_formats(formats, k["ALGO"] == "full")
    log.info(
        f"EXPORT_FORMATS={tuple(formats)} -> SAVE_FORMAT={save!r}, "
        f"SAVE_MERGED={merged}"
    )
    return {**k, "SAVE_FORMAT": save, "SAVE_MERGED": merged}


@cache
def _fused(path: str) -> bool:
    return stores_fused_qkv(path)


def fused_qkv(k: dict) -> bool:
    """SAVE_FUSED_QKV resolved (default False: separate q / k / v); `auto` writes q / k
    / v fused when the DiT file the run trains on stores them fused (the Comfy-Org
    files)."""
    choice = k["SAVE_FUSED_QKV"]
    if choice != "auto":
        return bool(choice)
    return _fused(str(Path(component(k, DIT)).resolve()))


def export_files(k: dict, fused: bool) -> list[ExportFile]:
    """The files one export writes: the adapters (ComfyUI LoRA / LyCORIS file) per
    SAVE_FORMAT kind, then with SAVE_MERGED the merged DiT (ComfyUI single file /
    diffusers folder); a full finetune writes the DiT files."""
    kinds = KINDS[k["SAVE_FORMAT"]]
    dit = {
        "comfy": ExportFile("merged", "merged", fused=fused),
        "hf": ExportFile("diffusers", "diffusers", folder=True),
    }
    if k["ALGO"] == "full":
        return [dit[kind] for kind in kinds]
    adapters = {
        "comfy": ExportFile("comfy", "comfy", fused=fused),
        "hf": ExportFile("lycoris", "lycoris"),
    }
    out = [adapters[kind] for kind in kinds]
    if k["SAVE_MERGED"]:
        out += [dit[kind] for kind in kinds]
    return out


def write_exports(
    files: list[ExportFile], trainable, backbone, root: Path, run: str, metadata: dict
) -> list[Path]:
    """Every file of `files` under `root` (created), the trainable's dW rows carrying
    the backbone's fold scales."""
    root.mkdir(parents=True, exist_ok=True)
    out = []
    scales = output_row_scales(backbone)
    for f in files:
        path = root / f.name(run)
        trainable.export(
            str(path), f.fmt, metadata, row_scales=scales, fused_qkv=f.fused
        )
        out.append(path)
    return out


def export_bytes(files: list[ExportFile], trainable, backbone) -> int:
    """About the bytes one export writes: an adapter file holds the trainable
    tensors, a DiT file the base parameters, in 16 bits."""
    adapter = sum(p.numel() for p in trainable.trainable_params()) * SAVE_BYTES
    dit = total_params(backbone.cfg) * SAVE_BYTES
    return sum(dit if f.fmt in ("merged", "diffusers") else adapter for f in files)


def log_export_plan(k: dict, files: list[ExportFile], size: int) -> None:
    """The build line: what each export writes and how large it is per checkpoint."""
    names = ", ".join(f.name(k["RUN_NAME"] or "run") for f in files) or "nothing"
    every = "each checkpoint and the end" if k["CHECKPOINT_EXPORT"] else "the end"
    log.info(
        f"exports (SAVE_FORMAT={k['SAVE_FORMAT']!r}) at {every}: {hl(names)}, about "
        f"{hl(f'{size / GIB:.2f} GiB')} each time"
    )
