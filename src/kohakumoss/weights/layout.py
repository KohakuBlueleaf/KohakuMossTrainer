"""Where each component's weights are. WEIGHTS is a ComfyUI models folder
(`diffusion_models/`, `text_encoders/`, `vae/`: one file per component, picked by name),
an HF pipeline folder (`transformer/`, `text_encoder/`, `tokenizer/`, `vae/`), or a
single DiT file; a component knob (TRANSFORMER / TEXT_ENCODER / VAE) is a path, or in a
ComfyUI folder a file name under the component's subfolder. See
docs/guides/weights.md."""

from pathlib import Path

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


def component_path(
    weights: str | Path,
    component: str,
    choice: str = "",
    variant: str = "base",
    full: bool = False,
) -> Path:
    """The weights of `component`: `choice` when it is an existing path; in a ComfyUI
    folder `comfy_file`; else `choice` as a path, the DiT at `weights` itself, the
    other components in `weights/<component>`."""
    if choice and Path(choice).exists():
        return Path(choice)
    if is_comfy_folder(weights):
        return comfy_file(weights, component, choice, variant, full)
    if choice:
        return Path(choice)
    return Path(weights) if component == DIT else Path(weights) / component


def layout_of(weights: str | Path) -> str:
    """`comfy` (a ComfyUI models folder), `file` (one safetensors file) or `hf`."""
    if is_comfy_folder(weights):
        return "comfy"
    return "file" if Path(weights).is_file() else "hf"
