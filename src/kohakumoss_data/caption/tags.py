"""Tag-string handling: splitting, unescaping, normalising."""

import re

ESCAPE = re.compile(r"\\([()\[\]{}])")


def split_tags(text: str, sep: str = ",") -> list[str]:
    """Split a tag string on `sep`, stripping blanks; order is kept."""
    return [t.strip() for t in text.split(sep) if t.strip()]


def unescape(tag: str) -> str:
    """`\\(` -> `(` and friends (webui-style escaped brackets)."""
    return ESCAPE.sub(r"\1", tag)


def _keep(tag: str) -> str:
    return tag


def _spaces(tag: str) -> str:
    return tag.replace("_", " ")


def _spaces_unescape(tag: str) -> str:
    return unescape(tag.replace("_", " "))


NORMALIZERS = {
    (False, False): _keep,
    (True, False): _spaces,
    (False, True): unescape,
    (True, True): _spaces_unescape,
}


def normalizer(underscores: bool, do_unescape: bool):
    """The tag normaliser for (underscore -> space, unescape)."""
    return NORMALIZERS[(bool(underscores), bool(do_unescape))]
