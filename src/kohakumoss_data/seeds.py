"""Counter-based seeding: every random decision is a pure function of its coordinates."""

import hashlib

import numpy as np


def seed_of(*parts) -> int:
    """64-bit seed from any tuple of ints / strings (blake2b of their repr)."""
    digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def rng_of(*parts) -> np.random.Generator:
    """A fresh PCG64 generator for the coordinates `parts`."""
    return np.random.Generator(np.random.PCG64(seed_of(*parts)))


def content_hash(data: bytes) -> str:
    """Hex blake2b-128 of raw bytes (cache keys, shard dedupe)."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def text_hash(*parts) -> str:
    """Hex blake2b-128 of a tuple of strings (cache keys)."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()
