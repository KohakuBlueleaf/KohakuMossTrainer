"""Helpers for the data scripts: knobs -> DataConfig, encoders from dotted factories."""

import importlib

from kohakumoss_data.config import DataConfig


def import_object(path: str):
    """`pkg.module.attr` or `pkg.module:attr` -> the attribute."""
    module, _, attr = path.replace(":", ".").rpartition(".")
    return getattr(importlib.import_module(module), attr)


def data_config(knobs: dict, **overrides) -> DataConfig:
    """DataConfig from UPPER_CASE globals, with field overrides."""
    cfg = DataConfig.from_knobs(knobs)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def build_encoder(spec: str, kwargs: dict):
    """An EncodeImages / EncodeText callable from a dotted factory path."""
    return import_object(spec)(**kwargs)
