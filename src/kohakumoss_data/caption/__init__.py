"""Caption rendering: tag handling, json->text, and the registered templates."""

from kohakumoss_data.caption.gbc import GbcTemplate
from kohakumoss_data.caption.tags import split_tags, unescape
from kohakumoss_data.caption.template import (
    JsonTemplate,
    MixTemplate,
    NaturalTemplate,
    TagTemplate,
    build_template,
)

__all__ = [
    "GbcTemplate",
    "JsonTemplate",
    "MixTemplate",
    "NaturalTemplate",
    "TagTemplate",
    "build_template",
    "split_tags",
    "unescape",
]
