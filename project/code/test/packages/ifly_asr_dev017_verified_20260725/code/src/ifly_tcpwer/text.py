"""Text normalization matching the challenge's SegLST token convention."""

from __future__ import annotations

import re
import unicodedata


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def normalize_token(text: str) -> list[str]:
    """Return Chinese characters and lowercase Latin words as scoring tokens."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    tokens: list[str] = []
    latin = ""
    for char in text:
        if _CJK.fullmatch(char):
            if latin:
                tokens.extend(_LATIN_TOKEN.findall(latin))
                latin = ""
            tokens.append(char)
        elif char.isascii() and (char.isalnum() or char in "'-"):
            latin += char
        else:
            if latin:
                tokens.extend(_LATIN_TOKEN.findall(latin))
                latin = ""
    if latin:
        tokens.extend(_LATIN_TOKEN.findall(latin))
    return tokens


def normalize_words(text: str) -> str:
    return " ".join(normalize_token(text))
