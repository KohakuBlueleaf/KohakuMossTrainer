"""Helpers for the data scripts: knobs -> DataConfig, encoders from dotted factories."""

from kohakumoss_common.imports import import_object
from kohakumoss_data.config import DataConfig

__all__ = ["build_encoder", "data_config", "import_object"]


def data_config(knobs: dict, **overrides) -> DataConfig:
    """DataConfig from UPPER_CASE globals with field overrides; the cache dir is resolved
    after the overrides apply."""
    upper = {k.upper(): v for k, v in overrides.items()}
    return DataConfig.from_knobs({**knobs, **upper})


def build_encoder(path: str, kwargs: dict):
    """An EncodeImages / EncodeText callable from a dotted factory path; the factory is
    called with `kwargs` and must return the callable itself."""
    encoder = import_object(path)(**kwargs)
    if not callable(encoder):
        raise TypeError(f"{path} returned a non-callable {type(encoder).__name__}")
    return encoder
