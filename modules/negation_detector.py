"""
Module: negation_detector.py
Scans a token window before each match for negation indicators.
Tags every match as affirmed=True or affirmed=False.
Works on both ExactMatch and SemanticMatch objects.
"""

import re
from typing import List, Union
from modules.normalizer import tokenize
from config.settings import settings


NEGATION_TERMS = {
    "no", "not", "never", "without", "none", "neither",
    "absence", "absent", "deny", "denies", "denied",
    "no history", "no known", "no prior", "no evidence",
    "not applicable", "n/a", "free of", "clear of",
    "has not", "have not", "had not", "does not", "did not",
}

# Patterns that negate even across a few words
NEGATION_PHRASES = [
    r"\bno\s+(known\s+)?(history|evidence|record|prior|previous)\s+of\b",
    r"\bwithout\s+(any\s+)?(history|prior|previous)\b",
    r"\bnot\s+(been\s+)?(involved|subject|party)\b",
    r"\bfree\s+of\b",
    r"\bclear\s+of\b",
    r"\bdenies?\s+(any|all)?\b",
    r"\bno\s+(outstanding|pending|active)\b",
]


def detect_negation(text: str, keyword: str, window: int = None) -> bool:
    """
    Returns True if the keyword occurrence in text appears to be negated.
    Checks both token-window and phrase patterns.
    """
    window = window or settings.negation_window_tokens
    normalized = text.lower()

    # Phrase-level negation check (higher precision)
    for pattern in NEGATION_PHRASES:
        if re.search(pattern, normalized):
            return True

    # Token window check around keyword position
    kw_lower = keyword.lower()
    kw_pos = normalized.find(kw_lower)
    if kw_pos == -1:
        return False

    tokens = tokenize(normalized[:kw_pos])
    window_tokens = tokens[-window:]

    for token in window_tokens:
        if token in NEGATION_TERMS:
            return True

    return False


def apply_negation_detection(matches: list, chunks_map: dict) -> list:
    """
    Applies negation detection to a list of match objects (Exact or Semantic).
    chunks_map: {chunk_index: chunk_text} for quick lookup
    Returns matches with .affirmed attribute set.
    """
    for match in matches:
        text = chunks_map.get(match.chunk_index, match.chunk_text)
        keyword = getattr(match, "keyword", getattr(match, "matched_keyword", ""))
        is_negated = detect_negation(text, keyword)
        match.affirmed = not is_negated
        match.negation_flag = is_negated

    return matches
