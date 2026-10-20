"""FLUX.2 VAE wrapper: pixels <-> normalised 128-channel /16 latents.

Encode: VAE posterior (mode or sample) [B, 32, H/8, W/8] -> 2x2 patchify -> 128 ch ->
(z - running_mean) / sqrt(running_var + eps). Decode is the exact inverse.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from torch import Tensor, nn

from kohakumoss.encoders.specs import SpecEncoder, vae_spec
from kohakumoss.encoders.vae.comfy import load_vae_file
from kohakumoss.offload.encoders.vae import place_vae
from kohakumoss.weights import VAE, comfy_file, hf_root, is_comfy_folder
from kohakumoss_kernels import compile_setup
from kohakumoss_kernels.dtypes import BY_NAME
from kohakumoss_kernels.hardware.arch import arch_of


def patchify(z: Tensor) -> Tensor:
    """[B, C, 2h, 2w] -> [B, 4C, h, w]; channel = c * 4 + i * 2 + j for pixel (2y+i, 2x+j)."""
    b, c, hh, ww = z.shape
    z = z.reshape(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return z.reshape(b, c * 4, hh // 2, ww // 2)


def unpatchify(z: Tensor) -> Tensor:
    """Inverse of `patchify`."""
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(b, c // 4, 2 * h, 2 * w)


DTYPES = {name: BY_NAME[name] for name in ("fp32", "bf16", "fp16")}
VAE_DTYPES = ("auto", "fp32", "bf16", "fp16")
FP16_PROBLEM: str | None = None
"""Why fp16 is not a VAE dtype; None: fp16 measured finite on real images at 1-2 MP.
See docs/internals/offload.md#vae."""
BATCH_PIXELS = 4 * 1024 * 1024


def _has_bf16(device: torch.device) -> bool:
    """bf16 on `device`: a CUDA card that has it (or one this process cannot query),
    XPU, and the host (natively, or through `host_gemm`'s fp32-accumulated 16-bit
    GEMMs)."""
    match device.type:
        case "cuda":
            return not torch.cuda.is_available() or arch_of(device).has_bf16
    return True


def resolve_vae_dtype(name: str | torch.dtype, device) -> torch.dtype:
    """VAE_DTYPE -> torch dtype for `device`, the device the latents and decodes are
    for (whatever the VAE's placement): `auto` is bf16 where the device has bf16, else
    fp16 when fp16 is safe, else fp32."""
    if isinstance(name, torch.dtype):
        return name
    if name != "auto":
        return DTYPES[name]
    device = torch.device(device)
    if _has_bf16(device):
        return torch.bfloat16
    if device.type != "cpu" and FP16_PROBLEM is None:
        return torch.float16
    return torch.float32


def vae_encoder_spec(
    path: str | Path = "", device: torch.device | str = "cuda", dtype="auto", **_
) -> str:
    """The `encoder_spec` `vae_encoder(...)` gives for the same arguments, without
    loading the weights."""
    return vae_spec(resolve_vae_dtype(dtype, device))


def vae_dtype_problems(name: str, device) -> list[str]:
    """Every problem with VAE_DTYPE=`name` on `device`."""
    if name not in VAE_DTYPES:
        return [f"VAE_DTYPE={name!r} not in {VAE_DTYPES}"]
    device = torch.device(device)
    if name == "fp16" and FP16_PROBLEM is not None:
        return [f"VAE_DTYPE='fp16': {FP16_PROBLEM}; use 'auto', 'bf16' or 'fp32'"]
    if name == "bf16" and device.type != "cpu" and not _has_bf16(device):
        return [f"VAE_DTYPE='bf16': {device} has no bf16; use 'auto' or 'fp32'"]
    return []


def vae_encoder(
    path: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | str = "auto",
    mode: str = "mode",
    batch: int = 1,
    compile: bool | None = None,
    plan: dict | None = None,
):
    """EncodeImages factory for the latent cache: `encode_images` of a FluxVae loaded
    from `path` (dtype a VAE_DTYPE name or a torch dtype) and placed by `plan` (VaePlan
    fields; memory default "auto": the fastest placement that fits), `batch` same-size
    images per encode call, `compile` channels_last + torch.compile (None: the plan's
    rule), carrying the VAE's `encoder_spec`."""
    if not str(path):
        raise ValueError(
            "vae_encoder: path is empty; set IMAGE_ENCODER_KWARGS['path'] to the VAE dir, "
            "a VAE file or a pipeline root"
        )
    resolved = resolve_vae_dtype(dtype, device)
    spec = {"memory": "auto", "batch": batch, **(plan or {})}
    if compile is not None:
        spec["compile"] = "on" if compile else "off"
    vae = FluxVae.from_pretrained(path, "cpu", resolved, mode)
    placed = place_vae(vae, spec, device)
    return SpecEncoder(placed.encode_images, vae_encoder_spec(path, device, resolved))


def to_pixels(image: np.ndarray, device, dtype) -> Tensor:
    """uint8 HWC (RGB) -> [1, 3, H, W] in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    return x.permute(2, 0, 1).unsqueeze(0).to(dtype) / 127.5 - 1.0


class FluxVae(nn.Module):
    """Frozen VAE with the MOSS latent normalisation (fp32 statistics, read from the
    VAE's BatchNorm before any cast); `mode` is "mode" or "sample"."""

    def __init__(self, vae: AutoencoderKLFlux2, mode: str = "mode"):
        super().__init__()
        self.vae = vae.eval().requires_grad_(False)
        bn = vae.bn
        eps = vae.config.batch_norm_eps
        self.register_buffer("mean", bn.running_mean.detach().clone().float(), False)
        self.register_buffer(
            "std", (bn.running_var.detach().float() + eps).sqrt(), False
        )
        match mode:
            case "mode":
                self._draw = _mode
            case "sample":
                self._draw = _sample
            case _:
                raise ValueError(f"unknown VAE encode mode {mode!r}")
        self.batch = 1
        self.batch_pixels = BATCH_PIXELS
        self._layout = _same
        self._encode = self.vae.encode
        self._decode = self.vae.decode

    def accelerate(self, channels_last: bool = False, compile: bool = False) -> None:
        """Run encode / decode in channels_last and / or through torch.compile."""
        if channels_last:
            self.vae.to(memory_format=torch.channels_last)
            self._layout = _channels_last
        if compile:
            compile_setup.apply()
            self._encode = torch.compile(self.vae.encode, dynamic=True)
            self._decode = torch.compile(self.vae.decode, dynamic=True)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        mode: str = "mode",
    ) -> "FluxVae":
        """`path` is the VAE dir, a pipeline root containing `vae/`, an HF weights root
        (its `base/` VAE), one safetensors file (diffusers or BFL keys, e.g. ComfyUI's
        `flux2-vae_small` / `flux2-vae`) or a ComfyUI models folder (its default VAE
        file). Loaded in fp32 (the
        normalisation statistics stay fp32), then the VAE is cast to `dtype`."""
        path = Path(path)
        path = comfy_file(path, VAE) if is_comfy_folder(path) else hf_root(path, VAE)
        if path.is_file():
            vae = load_vae_file(path)
        else:
            sub = "vae" if (path / "vae" / "config.json").is_file() else None
            vae = AutoencoderKLFlux2.from_pretrained(
                str(path), subfolder=sub, torch_dtype=torch.float32
            )
        out = cls(vae, mode)
        out.vae.to(dtype)
        return out.to(device)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    @property
    def encoder_spec(self) -> str:
        """The compute spec a latent cache key records for this VAE."""
        return vae_spec(self.dtype)

    @torch.no_grad()
    def encode(
        self, pixels: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """pixels [B, 3, H, W] in [-1, 1] (H, W multiples of 16) -> fp32 [B, 128, H/16, W/16]."""
        dist = self._encode(self._layout(pixels.to(self.dtype))).latent_dist
        z = patchify(self._draw(dist, generator).float())
        return (z - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """fp32 [B, 128, h, w] -> pixels [B, 3, 16h, 16w] in [-1, 1]."""
        z = latents.float() * self.std.view(1, -1, 1, 1) + self.mean.view(1, -1, 1, 1)
        z = self._layout(unpatchify(z).to(self.dtype))
        out = self._decode(z, return_dict=False)[0]
        return out.float().clamp(-1, 1)

    def encode_images(
        self, images: Sequence[np.ndarray], hw: Tensor, generator=None
    ) -> Tensor:
        """EncodeImages: uint8 HWC images sized 16*hw -> packed fp32 latents [N_img, 128].
        Runs of same-size images are encoded together, `batch` images and at most
        `batch_pixels` pixels per call."""
        sizes = [tuple(s) for s in hw.tolist()]
        for image, (h, w) in zip(images, sizes):
            if image.shape[0] != 16 * h or image.shape[1] != 16 * w:
                raise ValueError(f"image {image.shape[:2]} is not 16 * {(h, w)}")
        device = self.mean.device
        out = []
        for lo, hi in _runs(sizes, self.batch, self.batch_pixels):
            pixels = torch.cat(
                [to_pixels(i, device, self.dtype) for i in images[lo:hi]]
            )
            z = self.encode(pixels, generator)
            out += [zi.flatten(1).transpose(0, 1) for zi in z]
        return torch.cat(out)


def _runs(sizes: list[tuple], batch: int, batch_pixels: int) -> list[tuple[int, int]]:
    """[lo, hi) ranges of consecutive equal sizes, each at most `batch` images and
    `batch_pixels` pixels (always at least one image)."""
    out, lo = [], 0
    while lo < len(sizes):
        h, w = sizes[lo]
        cap = max(1, min(batch, batch_pixels // max(1, 256 * h * w)))
        hi = lo + 1
        while hi < len(sizes) and hi - lo < cap and sizes[hi] == sizes[lo]:
            hi += 1
        out.append((lo, hi))
        lo = hi
    return out


def _same(x: Tensor) -> Tensor:
    return x


def _channels_last(x: Tensor) -> Tensor:
    return x.contiguous(memory_format=torch.channels_last)


def _mode(dist, generator):
    return dist.mode()


def _sample(dist, generator):
    return dist.sample(generator)
