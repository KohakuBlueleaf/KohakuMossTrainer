"""Integer helpers of the kernels (ceil division, power-of-two buckets) and the
problem-key buckets of the packed text-encoder ops (varlen delta rule and causal
attention), shared by the kernels' keys and the prewarm that covers them. See
docs/internals/tuning.md#text-encoder-buckets."""

TOKEN_FLOOR = 256  # smallest token bucket of the block-op and adapter problem keys
LENGTH_FLOOR = 64
MEAN_FLOOR = 64
SEGMENT_CAP = 8
KEY_VERSION = "k2"


def cdiv(a: int, b: int) -> int:
    """ceil(a / b) for positive `b`."""
    return -(-a // b)


def pow2(n: int, floor: int) -> int:
    """Smallest power of two >= max(n, floor) (`floor` a power of two)."""
    b = floor
    while b < n:
        b *= 2
    return b


def chunk_rows(n: int, rows: int, sms: int, per_sm: int) -> int:
    """Rows per segment tile of `n` rows: about `per_sm` tiles per SM over `sms` SMs, a
    multiple of `rows`."""
    return max(rows, cdiv(cdiv(n, per_sm * sms), rows) * rows)


def pow2_buckets(top: int, floor: int = TOKEN_FLOOR) -> list[int]:
    """One count per power-of-two bucket from `floor` up to `top`'s: the bucket's top,
    or `top` when that is smaller."""
    out, n = [], floor
    while True:
        out.append(min(n, top))
        if n >= top:
            return out
        n *= 2


def length_bucket(max_len: int) -> int:
    """Bucket of the longest segment."""
    return pow2(max_len, LENGTH_FLOOR)


def mean_bucket(tokens: int, segments: int) -> int:
    """Bucket of the mean segment length."""
    return pow2(tokens // max(segments, 1), MEAN_FLOOR)


def segment_bucket(segments: int) -> int:
    """Bucket of the segment count: 1, 2, 4, then `SEGMENT_CAP` and above as one."""
    return min(pow2(segments, 1), SEGMENT_CAP)


def delta_buckets(tokens: int, segments: int, max_len: int) -> tuple[int, int, int]:
    """(segments, longest, mean) buckets of a delta-rule problem."""
    return (
        segment_bucket(segments),
        length_bucket(max_len),
        mean_bucket(tokens, segments),
    )
