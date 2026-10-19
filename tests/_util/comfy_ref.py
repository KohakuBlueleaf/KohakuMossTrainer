"""Independent numpy references of the ComfyUI quant formulas (comfy-kitchen's eager
backend: `backends/eager/quantization.py`, `backends/eager/w4a8_int8.py`), in fp64:
the regular Hadamard, the ConvRot unrotation, the W4A8 code packing and decode."""

import numpy as np

H4 = np.array(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64
)
LUT = np.array(
    [
        -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210,
        -0.050720, 0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953,
        0.794876, 0.980671,
    ],
    dtype=np.float32,
)  # fmt: skip


def hadamard_np(group: int) -> np.ndarray:
    """The normalised regular Hadamard of size `group` (a power of 4): kron of H4."""
    h = H4
    while h.shape[0] < group:
        h = np.kron(h, H4)
    return h / np.sqrt(group)


def unrotate_np(w_rot: np.ndarray, group: int) -> np.ndarray:
    """W = W_rot . blockdiag(H) over the last dim (H symmetric and involutory)."""
    shape = w_rot.shape
    return (w_rot.astype(np.float64).reshape(-1, group) @ hadamard_np(group)).reshape(
        shape
    )


def pack_np(codes: np.ndarray, bits: int) -> np.ndarray:
    """Unsigned codes [N, K] -> the int8 storage: the K/2 nibble plane (even column =
    low nibble), then at 6 bits the K/4 plane of each code's top two bits."""
    codes = codes.astype(np.int64)
    low = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    planes = [low]
    if bits == 6:
        high = np.zeros((codes.shape[0], codes.shape[1] // 4), dtype=np.int64)
        for j in range(4):
            high |= ((codes[:, j::4] >> 4) & 0x3) << (2 * j)
        planes.append(high)
    return np.concatenate(planes, axis=1).astype(np.uint8).view(np.int8)


def decode_np(
    packed: np.ndarray,
    s_rel: np.ndarray,
    codebook: np.ndarray | None,
    group: int,
    bits: int,
) -> np.ndarray:
    """The int8 grid of packed codes: round(level * s_rel) clamped to +-127, the level
    codebook[code] or code - 2^(bits-1)."""
    n = packed.shape[0]
    k = s_rel.shape[1] * group
    raw = packed.view(np.uint8).astype(np.int64)
    codes = np.zeros((n, k), dtype=np.int64)
    for c in range(k):
        byte = raw[:, c // 2]
        codes[:, c] = (byte >> 4) & 0xF if c % 2 else byte & 0xF
        if bits == 6:
            top = raw[:, k // 2 + c // 4]
            codes[:, c] |= ((top >> (2 * (c % 4))) & 0x3) << 4
    if codebook is not None:
        levels = codebook.astype(np.float32)[codes]
    else:
        levels = (codes - (1 << (bits - 1))).astype(np.float32)
    scale = np.repeat(s_rel.astype(np.float32), group, axis=1)
    return np.clip(np.rint(levels * scale), -127, 127).astype(np.int8)


def w4a8_weight_np(
    grid: np.ndarray,
    s_channel: np.ndarray,
    group: int,
    rotation: int,
    correction: np.ndarray | None = None,
) -> np.ndarray:
    """The floating weight of an int8 grid: grid * s_channel (+ correction per group
    and row), unrotated."""
    w = grid.astype(np.float64) * s_channel.astype(np.float64)[:, None]
    if correction is not None:
        w = w + np.repeat(correction.T.astype(np.float64), group, axis=1)
    return unrotate_np(w, rotation) if rotation else w
