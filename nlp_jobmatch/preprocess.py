"""Light text cleanup for classic NLP matching."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9+#\s-]")


def normalize(text: str) -> str:
    """Lowercase, drop most punctuation, collapse whitespace."""
    cleaned = _NON_WORD.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", cleaned).strip()


def word_count(text: str) -> int:
    return len(normalize(text).split()) if text.strip() else 0
