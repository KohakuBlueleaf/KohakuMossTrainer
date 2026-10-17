"""Helpers for the data scripts: knobs -> DataConfig, encoders from dotted factories."""

from kohakumoss_data.config import DataConfig
from kohakumoss_data.imports import import_object

__all__ = ["build_encoder", "data_config", "import_object"]


def data_config(knobs: dict, **overrides) -> DataConfig:
    """DataConfig from UPPER_CASE globals (cache dir resolved), with field overrides."""
    cfg = DataConfig.from_knobs(knobs)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def build_encoder(spec: str, kwargs: dict):
    """An EncodeImages / EncodeText callable from a dotted factory path; the factory is
    called with `kwargs` and must return the callable itself."""
    encoder = import_object(spec)(**kwargs)
    if not callable(encoder):
        raise TypeError(f"{spec} returned a non-callable {type(encoder).__name__}")
    return encoder
