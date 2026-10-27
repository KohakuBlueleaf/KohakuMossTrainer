"""Frozen W4A8 base weights (ComfyUI `asym_w4a8_int8`): 4-bit (or 6-bit) codes in
groups of `group` input columns decode through a 16-level `codebook` (or the uniform
grid) times the e4m3 group scale `s_rel` onto an int8 grid; times the per-row
`s_channel` (plus the per-group `correction` when stored) that is the rotated weight
W R (R = blockdiag(H_g) over K, ConvRot).

Two runs of the same int8 values: `to_int8()` decodes once into a QuantWeight (1 byte
per weight), `linear` keeps the packed codes and decodes per call before the rotated
int8 linear. See docs/internals/quant-kernels.md#w4a8-weights.
"""

import functools
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kohakumoss.quant.weights import QuantWeight, unrotated
from kohakumoss_kernels import compile_setup, registry
from kohakumoss_kernels.quant import quant_linear_eager
from kohakumoss_kernels.quant.rotated import quant_linear_rotated_eager

BITS = (4, 6)
COMPILED_KINDS = ("cuda", "xpu")


def geometry(q: Tensor, s_rel: Tensor, group: int) -> tuple[int, int, int]:
    """(N, K, bits) of packed codes [N, K * bits / 8] with group scales [N, K / group];
    ValueError for any other layout."""
    n, cols = q.shape
    k = s_rel.shape[1] * group
    bits = (cols * 8) // k if k > 0 else 0
    if (
        bits not in BITS
        or k * bits != cols * 8
        or tuple(s_rel.shape) != (n, k // group)
    ):
        raise ValueError(
            f"W4A8 codes {tuple(q.shape)} with group scales {tuple(s_rel.shape)} and "
            f"group {group} are not a 4- or 6-bit layout"
        )
    return n, k, bits


def unpack_codes(q: Tensor, k: int, bits: int) -> Tensor:
    """Unsigned int32 codes [N, K] of packed int8 rows: the K/2-byte nibble plane
    (even column = low nibble), then at 6 bits the K/4-byte plane of each code's top two
    bits (column c in byte c // 4 at bit 2 (c % 4))."""
    n = q.shape[0]
    packed = q.view(torch.uint8).to(torch.int32)
    low = packed[:, : k // 2]
    codes = torch.stack((low & 0xF, (low >> 4) & 0xF), dim=-1).reshape(n, k)
    if bits == 6:
        shifts = torch.arange(0, 8, 2, device=q.device, dtype=torch.int32)
        high = (packed[:, k // 2 :].unsqueeze(-1) >> shifts) & 0x3
        codes = codes | (high.reshape(n, k) << 4)
    return codes


def decode_codes(
    q: Tensor, s_rel: Tensor, codebook: Tensor | None, group: int, bits: int
) -> Tensor:
    """The int8 grid [N, K] of packed codes: round(level * s_rel) clamped to +-127, the
    level `codebook[code]` or (code - 2^(bits-1)) without a codebook."""
    k = s_rel.shape[1] * group
    codes = unpack_codes(q, k, bits)
    if codebook is not None:
        levels = codebook.float()[codes]
    else:
        levels = (codes - (1 << (bits - 1))).float()
    n = q.shape[0]
    values = levels.view(n, k // group, group) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp_(-127, 127).to(torch.int8)


@functools.cache
def _compiled_decode():
    compile_setup.apply()
    return torch.compile(decode_codes, dynamic=True)


def decoder_for(device: torch.device):
    """The decode the per-call path runs on `device`: compiled on CUDA / XPU, eager on
    the host."""
    return _compiled_decode() if device.type in COMPILED_KINDS else decode_codes


class W4A8Weight(nn.Module):
    """BaseWeight of packed W4A8 codes (buffers: `q`, `s_rel`, `s_channel`, and
    `codebook` / `correction` when stored); `rotation` is the ConvRot group (0: none).
    """

    fmt = "w4a8"

    def __init__(
        self,
        q: Tensor,
        s_rel: Tensor,
        s_channel: Tensor,
        codebook: Tensor | None,
        group: int,
        rotation: int = 0,
        correction: Tensor | None = None,
    ):
        super().__init__()
        n, k, bits = geometry(q, s_rel, group)
        if correction is not None and tuple(correction.shape) != (k // group, n):
            raise ValueError(
                f"W4A8 correction {tuple(correction.shape)} is not {(k // group, n)}"
            )
        self.group, self.bits, self.rotation = int(group), bits, int(rotation)
        self.register_buffer("q", q)
        self.register_buffer("s_rel", s_rel)
        self.register_buffer("s_channel", s_channel.float().reshape(-1))
        self.register_buffer("codebook", None if codebook is None else codebook.float())
        self.register_buffer("correction", correction)
        self.shape = torch.Size((n, k))
        self.decoder = decode_codes
        self.int8_dgrad: str | None = None
        self._linear = _eager_for(self)

    def decode(self) -> Tensor:
        """The int8 grid [N, K] of the codes (eager)."""
        return decode_codes(self.q, self.s_rel, self.codebook, self.group, self.bits)

    def rotated_weight(self) -> Tensor:
        """The stored (rotated) weight W R in fp32."""
        w = self.decode().float().view(self.shape[0], -1, self.group)
        w = w * self.s_channel.view(-1, 1, 1)
        if self.correction is not None:
            w = w + self.correction.t().float().unsqueeze(-1)
        return w.view(self.shape)

    def weight(self, dtype: torch.dtype) -> Tensor:
        """W in `dtype` (the rotation undone)."""
        return unrotated(self.rotated_weight(), self.rotation, dtype)

    def materialize(self) -> Tensor:
        return self.weight(torch.float32)

    @property
    def decodes_to_int8(self) -> bool:
        """The weight is exactly int8 codes times `s_channel` (no stored correction)."""
        return self.correction is None

    def to_int8(self) -> QuantWeight:
        """The QuantWeight of the decoded int8 grid and `s_channel` (same rotation)."""
        if not self.decodes_to_int8:
            raise ValueError("a W4A8 weight with a correction term has no int8 form")
        return QuantWeight(
            self.decode(), self.s_channel, "int8", self.rotation, self.int8_dgrad
        )

    def bind(
        self, dtype: torch.dtype, device: torch.device | str | None = None
    ) -> None:
        """Bind `linear` for (device, dtype) once: per call, the codes decoded
        (`decoder_for`) into the rotated int8 linear (`quant_linear_rotated`, eager
        when no kernel serves; with `int8_dgrad` set off the host, the `int8_linear`
        op); a stored correction runs the dense weight."""
        device = self.q.device if device is None else torch.device(device)
        self.decoder = decoder_for(device)
        self._linear = _eager_for(self)
        if not self.decodes_to_int8:
            return
        if self.int8_dgrad is not None and device.type != "cpu":
            kernel = registry.resolve(
                "int8_linear", device, dtype, group=self.rotation, dgrad=self.int8_dgrad
            )
            self._linear = partial(_int8_kernel, kernel)
            return
        name = "quant_linear_rotated" if self.rotation else "quant_linear"
        query = {"group": self.rotation} if self.rotation else {}
        try:
            kernel = registry.resolve(name, device, dtype, fmt="int8", **query)
        except (registry.NoImplementationError, KeyError):
            return
        run = _rotated_kernel if self.rotation else _plain_kernel
        self._linear = partial(run, kernel)

    def linear(self, x: Tensor, bias: Tensor | None) -> Tensor:
        """x W^T (+ bias) through the bound implementation."""
        return self._linear(self, x, bias)

    def codes(self) -> Tensor:
        """The int8 grid through the bound decoder."""
        return self.decoder(self.q, self.s_rel, self.codebook, self.group, self.bits)

    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.buffers())

    def int8_nbytes(self) -> int:
        """Bytes of the `to_int8()` form (codes and row scales)."""
        return self.shape.numel() + self.s_channel.nbytes


def _cast(bias: Tensor | None, x: Tensor) -> Tensor | None:
    return None if bias is None else bias.to(x.dtype)


def _eager_for(w: W4A8Weight):
    """The eager linear of `w`'s layout: the dense weight with a correction term,
    else the decoded codes through the (rotated) eager quant op."""
    if not w.decodes_to_int8:
        return _dense
    return _rotated_eager if w.rotation else _plain_eager


def _rotated_eager(w: W4A8Weight, x: Tensor, bias: Tensor | None) -> Tensor:
    return quant_linear_rotated_eager(
        x, w.codes(), w.s_channel, bias, "int8", w.rotation
    )


def _plain_eager(w: W4A8Weight, x: Tensor, bias: Tensor | None) -> Tensor:
    return quant_linear_eager(x, w.codes(), w.s_channel, bias, "int8")


def _dense(w: W4A8Weight, x: Tensor, bias: Tensor | None) -> Tensor:
    return F.linear(x, w.weight(x.dtype), _cast(bias, x))


def _rotated_kernel(kernel, w: W4A8Weight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, w.codes(), w.s_channel, bias, "int8", w.rotation)


def _plain_kernel(kernel, w: W4A8Weight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, w.codes(), w.s_channel, bias, "int8")


def _int8_kernel(kernel, w: W4A8Weight, x: Tensor, bias: Tensor | None) -> Tensor:
    return kernel(x, w.codes(), w.s_channel, bias, w.rotation, w.int8_dgrad)
