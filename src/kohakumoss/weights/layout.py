"""Where each component's weights are. WEIGHTS is a ComfyUI models folder
(`diffusion_models/`, `text_encoders/`, `vae/`: one file per component, picked by name),
an HF pipeline folder (`transformer/`, `text_encoder/`, `tokenizer/`, `vae/`), or a
single DiT file; a component knob (TRANSFORMER / TEXT_ENCODER / VAE) is a path, or in a
ComfyUI folder a file name under the component's subfolder. See
docs/guides/weights.md."""

from pathlib import Path

from kohakumoss_common.log import get_logger

log = get_logger(__name__)
DIT, TEXT, VAE = "transformer", "text_encoder", "vae"
COMPONENTS = (DIT, TEXT, VAE)
COMFY_DIRS = {DIT: "diffusion_models", TEXT: "text_encoders", VAE: "vae"}
SUFFIX = ".safetensors"
COMFY_DEFAULTS = {
    DIT: ("moss_image_{variant}_int8_convrot", "moss_image_{variant}_bf16"),
    TEXT: ("qwen3.5_9b_int8_convrot", "qwen3.5_9b_bf16", "qwen3.5_9b_w4a8_gptq"),
    VAE: ("flux2-vae_small", "flux2-vae"),
}
FULL_FT_DIT = ("moss_image_{variant}_bf16", "moss_image_{variant}_int8_convrot")


def is_comfy_folder(path: str | Path) -> bool:
    """`path` is a ComfyUI models folder: a directory holding `diffusion_models/`,
    `text_encoders/` or `vae/` with safetensors files and no HF `transformer/`."""
    path = Path(path)
    if not path.is_dir() or (path / DIT).is_dir():
        return False
    return any(any((path / sub).glob(f"*{SUFFIX}")) for sub in COMFY_DIRS.values())


def comfy_defaults(component: str, variant: str = "base", full: bool = False):
    """The file names a ComfyUI folder's `component` defaults to, in order: the int8
    ConvRot DiT and text encoder first (a full finetune takes the bf16 DiT first), the
    MOSS-Image VAE (`flux2-vae_small`) first."""
    names = FULL_FT_DIT if component == DIT and full else COMFY_DEFAULTS[component]
    return tuple(n.format(variant=variant) + SUFFIX for n in names)


def _named(folder: Path, component: str, name: str) -> Path:
    sub = folder / COMFY_DIRS[component]
    return sub / (name if name.endswith(SUFFIX) else name + SUFFIX)


def comfy_file(
    folder: str | Path,
    component: str,
    choice: str = "",
    variant: str = "base",
    full: bool = False,
) -> Path:
    """The file of `component` in ComfyUI models `folder`: `choice` (a file name, with
    or without `.safetensors`, under the component's subfolder), else the first default
    that exists (`comfy_defaults`); FileNotFoundError naming what was tried."""
    folder = Path(folder)
    names = (choice,) if choice else comfy_defaults(component, variant, full)
    tried = [_named(folder, component, n) for n in names]
    for path in tried:
        if path.is_file():
            return path
    listed = ", ".join(str(p) for p in tried)
    raise FileNotFoundError(f"no {component} file in {folder}: tried {listed}")


def ships(root: str | Path, name: str) -> bool:
    """`root/name/` holds safetensors files."""
    return any((Path(root) / name).glob(f"*{SUFFIX}"))


def component_root(path: str | Path, variant: str | None, name: str) -> Path:
    """The folder holding component `name` (a `name/` with safetensors files):
    `path/<variant>` (a weights root with base/ sft/); `path/base` when that variant
    folder does not ship `name`'s weights (logged); `path` itself (one pipeline dir)
    when there is no variant folder. FileNotFoundError when none holds it. See
    docs/guides/inference.md."""
    path = Path(path)
    own = path / variant if variant else None
    if own is not None and own.is_dir():
        for root in (own, path / "base"):
            if ships(root, name):
                if root != own:
                    log.info(f"{name}/ from {root} ({own} does not ship its weights)")
                return root
        raise FileNotFoundError(f"no {name}/ weights in {own} or {path / 'base'}")
    if ships(path, name):
        return path
    where = f"no VARIANT folder {own}/ and " if own is not None else ""
    raise FileNotFoundError(f"{where}no {name}/ under {path}")


def hf_root(path: str | Path, component: str, variant: str | None = "base") -> Path:
    """The HF pipeline folder of `component` under `path`: a weights root's variant
    folder (`component_root`), else `path` itself (a pipeline dir, a component dir or a
    file)."""
    path = Path(path)
    if not path.is_dir():
        return path
    try:
        return component_root(path, variant, component)
    except FileNotFoundError:
        return path


def component_path(
    weights: str | Path,
    component: str,
    choice: str = "",
    variant: str = "base",
    full: bool = False,
) -> Path:
    """The weights of `component`: `choice` when it is an existing path; in a ComfyUI
    folder `comfy_file`; else `choice` as a path, the DiT at the HF pipeline folder
    (`hf_root`: a weights root's `<variant>/`), the other components in its
    `<component>/`."""
    if choice and Path(choice).exists():
        return Path(choice)
    if is_comfy_folder(weights):
        return comfy_file(weights, component, choice, variant, full)
    if choice:
        return Path(choice)
    root = hf_root(weights, component, variant)
    return root if component == DIT else root / component


def layout_of(weights: str | Path) -> str:
    """`comfy` (a ComfyUI models folder), `file` (one safetensors file) or `hf`."""
    if is_comfy_folder(weights):
        return "comfy"
    return "file" if Path(weights).is_file() else "hf"
