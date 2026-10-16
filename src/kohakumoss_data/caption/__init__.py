"""Caption rendering: tag handling, json->text, and the registered templates."""

from kohakumoss_data.caption.json_text import json_to_text
from kohakumoss_data.caption.tags import split_tags, unescape
from kohakumoss_data.caption.template import (
    JsonTemplate,
    MixTemplate,
    NaturalTemplate,
    TagTemplate,
    build_template,
)

__all__ = [
    "JsonTemplate",
    "MixTemplate",
    "NaturalTemplate",
    "TagTemplate",
    "build_template",
    "json_to_text",
    "split_tags",
    "unescape",
]
