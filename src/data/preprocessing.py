"""
Text Preprocessing Module for NLP Persistent Memory Pipeline
Provides robust cleaning, normalization, duplicate handling, and character sanitization.
"""

import re
import unicodedata
from typing import Optional, List, Set


def normalize_whitespace(text: str) -> str:
    """Replaces redundant whitespace, newlines, and tabs with single spaces or standard newlines."""
    if not text:
        return ""
    # Normalize unicode representations
    text = unicodedata.normalize("NFKC", text)
    # Strip carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Collapse multiple inline spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_text(text: Optional[str]) -> str:
    """General text cleaning for ingestion into vector stores and classifiers."""
    if text is None:
        return ""
    # Strip null bytes and non-printable control characters (except newline, tab)
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return normalize_whitespace(cleaned)


def is_valid_text(text: Optional[str], min_length: int = 5) -> bool:
    """Checks if text meets minimum content validity thresholds."""
    if not text:
        return False
    cleaned = clean_text(text)
    return len(cleaned) >= min_length


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Preserves order while removing exact normalized duplicates."""
    seen: Set[str] = set()
    unique: List[str] = []
    for t in texts:
        norm = clean_text(t)
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(t)
    return unique
