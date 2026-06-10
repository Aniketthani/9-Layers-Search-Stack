"""
Module: normalizer.py
Dynamic linguistic normalization pipeline.

Approach: for each keyword word from the dictionary, build ONE regex
that matches the word plus its most common English inflections.
The regex is anchored with word boundaries so it only matches whole words.

This is reliable because:
  - We match forward FROM the known base form (not backward from variants)
  - We never try to "stem" words — we only extend them with known suffixes
  - The base form from the dictionary is always the canonical form
  - No stem extraction needed — the keyword itself is the anchor

Example: keyword "litigation"
  Generated regex: \blitigation(?:s|ed|ing|al|ally)?\b
  Matches: litigation, litigations  (but NOT litigated — separate rule below)

For words ending in known morpheme boundaries, we also add cross-form patterns:
  "litigate" → also catches litigated, litigating, litigator
  "contamination" → also catches contaminate, contaminated, contaminating
"""

import re
from typing import List, Dict, Tuple


# ── Static: insurance abbreviation expansion ───────────────────────────────────
ABBREVIATIONS = {
    r"\bUST\b":      "underground storage tank",
    r"\bAST\b":      "aboveground storage tank",
    r"\bE&O\b":      "errors and omissions",
    r"\bGL\b":       "general liability",
    r"\bWC\b":       "workers compensation",
    r"\bBOP\b":      "business owners policy",
    r"\bCGL\b":      "commercial general liability",
    r"\bD&O\b":      "directors and officers",
    r"\bE&S\b":      "excess and surplus",
    r"\bMGA\b":      "managing general agent",
    r"\bPML\b":      "probable maximum loss",
    r"\bTIV\b":      "total insured value",
    r"\blitig\.":    "litigation",
    r"\benv\.":      "environmental",
    r"\brem\.":      "remediation",
    r"\bcontam\.":   "contamination",
    r"\bprop\.":     "property",
    r"\binsur\.":    "insurance",
    r"\bclaims?\.":  "claims",
    r"\bbankr\.":    "bankruptcy",
    r"\bregul\.":    "regulatory",
    r"\bocc\.":      "occupancy",
    r"\bviol\.":     "violation",
    r"\bstruc\.":    "structural",
    r"\bdmg\.":      "damage",
    r"\bhaz\.":      "hazardous",
}

FORMATTING_PATTERNS = [
    (r"[\u2019\u2018\u201c\u201d\u2014\u2013]", " "),
    (r"\s*-\s*",   " "),
    (r"-{2,}",     " "),
    (r"[^\w\s\.\,\!\?\;\:\(\)\/]", " "),
    (r"\s{2,}",    " "),
]

SKIP_WORDS = {
    "the","and","for","are","but","not","you","all","can","had","her","was",
    "one","our","out","day","get","has","him","his","how","its","let","man",
    "new","now","old","see","two","way","who","did","oil","sit","set","use",
    "fire","mold","lead","loss","lien","deed","debt","fine","fee","tax","act",
    "law","code","suit","zone","site","risk","plan","case","type","form","area",
    "work","land","home","void","null","none","some","with","from","into","over",
    "been","also","than","that","this","they","will","have","said","each","when",
    "high","upon","then","both","here","more","only","very","even","know","make",
}


def _word_variants_regex(word: str) -> Tuple[str, str]:
    """
    Given a base keyword word, return (pattern, canonical) where:
    - pattern is a regex matching the word and its common inflections
    - canonical is the form to normalize to (always the input word)

    Strategy: match the word as given, then add suffix alternations based
    on how the word ends. We look forward from the known form — never backward.
    """
    w = word.lower()

    # Words under 4 chars: match exactly, no inflection
    if len(w) < 4:
        return (r"\b" + re.escape(w) + r"\b", w)

    # ── -tion / -sion endings ─────────────────────────────────────────────────
    # litigation → litigations, litigate, litigated, litigating, litigator
    # contamination → contaminations, contaminate, contaminated, contaminating
    if w.endswith("tion") or w.endswith("sion"):
        base = w[:-3]  # strip 'ion': litigation → litigat, revision → revis
        pattern = (
            r"\b(?:"
            + re.escape(w) + r"s?"            # litigation, litigations
            + r"|" + re.escape(base) + r"e"   # litigate
            + r"|" + re.escape(base) + r"es"  # litigates
            + r"|" + re.escape(base) + r"ed"  # litigated
            + r"|" + re.escape(base) + r"ing" # litigating
            + r"|" + re.escape(base) + r"or"  # litigator
            + r"|" + re.escape(base) + r"ors" # litigators
            + r"|" + re.escape(base) + r"ory" # litigatory
            + r")\b"
        )
        return (pattern, w)

    # ── -ment endings ─────────────────────────────────────────────────────────
    # settlement → settlements, settle, settled, settling, settler
    if w.endswith("ment"):
        base = w[:-4]  # settl(e)ment → settle... keep trailing e if present
        pattern = (
            r"\b(?:"
            + re.escape(w) + r"s?"
            + r"|" + re.escape(base) + r"e?"
            + r"|" + re.escape(base) + r"e?s"
            + r"|" + re.escape(base) + r"ed"
            + r"|" + re.escape(base) + r"ing"
            + r"|" + re.escape(base) + r"er"
            + r"|" + re.escape(base) + r"ers"
            + r")\b"
        )
        return (pattern, w)

    # ── -ure endings ──────────────────────────────────────────────────────────
    # foreclosure → foreclosures, foreclose, foreclosed, foreclosing
    if w.endswith("ure"):
        base = w[:-3]  # foreclosure → foreclos
        pattern = (
            r"\b(?:"
            + re.escape(w) + r"s?"
            + r"|" + re.escape(base) + r"e"
            + r"|" + re.escape(base) + r"es"
            + r"|" + re.escape(base) + r"ed"
            + r"|" + re.escape(base) + r"ing"
            + r")\b"
        )
        return (pattern, w)

    # ── -tion/-acy/-ancy/-ency/-ity/-ance/-ence → base is canonical ──────────
    # bankruptcy → bankruptcies  (y→ies)
    if w.endswith("cy") or w.endswith("sy"):
        base = w[:-1]  # bankruptcy → bankrupt c → bankruptc
        pattern = (
            r"\b(?:"
            + re.escape(w)
            + r"|" + re.escape(w[:-1]) + r"ies"  # bankruptcies
            + r")\b"
        )
        return (pattern, w)

    # ── -al endings ───────────────────────────────────────────────────────────
    # structural → structurally, structure, structures
    if w.endswith("al"):
        pattern = (
            r"\b(?:"
            + re.escape(w)
            + r"|" + re.escape(w) + r"ly"
            + r"|" + re.escape(w[:-2]) + r"e"
            + r"|" + re.escape(w[:-2]) + r"es"
            + r"|" + re.escape(w[:-2]) + r"ed"
            + r"|" + re.escape(w[:-2]) + r"ing"
            + r")\b"
        )
        return (pattern, w)

    # ── -ous endings ──────────────────────────────────────────────────────────
    # hazardous → hazard
    if w.endswith("ous"):
        base = w[:-3]
        pattern = (
            r"\b(?:"
            + re.escape(w)
            + r"|" + re.escape(base)
            + r"|" + re.escape(base) + r"s"
            + r")\b"
        )
        return (pattern, w)

    # ── -y endings ────────────────────────────────────────────────────────────
    # remedy → remedied, remedies, remedying
    if w.endswith("y") and len(w) > 4:
        base = w[:-1]
        pattern = (
            r"\b(?:"
            + re.escape(w)
            + r"|" + re.escape(base) + r"ies"
            + r"|" + re.escape(base) + r"ied"
            + r"|" + re.escape(base) + r"ying"
            + r")\b"
        )
        return (pattern, w)

    # ── -e endings ────────────────────────────────────────────────────────────
    # litigate → litigated, litigating, litigator, litigates, litigation
    if w.endswith("e") and len(w) > 4:
        base = w[:-1]
        pattern = (
            r"\b(?:"
            + re.escape(w) + r"s?"
            + r"|" + re.escape(base) + r"ed"
            + r"|" + re.escape(base) + r"ing"
            + r"|" + re.escape(base) + r"or"
            + r"|" + re.escape(base) + r"ors"
            + r"|" + re.escape(base) + r"ion"
            + r"|" + re.escape(base) + r"ions"
            + r")\b"
        )
        return (pattern, w)

    # ── Default: plain noun/verb with common endings ───────────────────────────
    # lawsuit → lawsuits
    # flood → flooded, flooding, floods
    # damage → damages, damaged, damaging
    last = w[-1]
    if last in "aeiou":
        # ends in vowel (other than -e, handled above)
        pattern = r"\b" + re.escape(w) + r"s?\b"
    else:
        # ends in consonant — add common verb/noun endings
        # handle doubled consonant for short words (flood→flooded)
        doubled = w + last  # floodd
        pattern = (
            r"\b(?:"
            + re.escape(w) + r"s?"
            + r"|" + re.escape(w) + r"es"
            + r"|" + re.escape(w) + r"ed"
            + r"|" + re.escape(w) + r"ing"
            + r"|" + re.escape(w) + r"er"
            + r"|" + re.escape(w) + r"ers"
            + r"|" + re.escape(doubled) + r"ed"
            + r"|" + re.escape(doubled) + r"ing"
            + r"|" + re.escape(doubled) + r"er"
            + r")\b"
        )

    return (pattern, w)


def build_dynamic_morph_patterns(keywords_dict: dict) -> List[Tuple[re.Pattern, str]]:
    """
    Build compiled morphological patterns from the keyword dictionary.

    For every significant word in every keyword, generates a regex that
    matches that word plus its common inflected and derived forms.
    All matched forms are normalized to the canonical (dictionary) form.

    Returns list of (compiled_pattern, canonical_word) tuples.
    Cached per dictionary version — rebuilt only when content changes.
    """
    # Track which words we've already handled to avoid duplicates
    processed: Dict[str, str] = {}  # word → canonical

    for category, keywords in keywords_dict.items():
        for kw in keywords:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", kw.lower())
            for word in words:
                if word not in SKIP_WORDS and word not in processed:
                    processed[word] = word  # canonical = the dict word itself

    compiled = []
    for word, canonical in processed.items():
        pattern_str, canon = _word_variants_regex(word)
        try:
            compiled.append((re.compile(pattern_str, re.IGNORECASE), canon))
        except re.error:
            pass

    return compiled


_PATTERN_CACHE: Dict[str, List[Tuple[re.Pattern, str]]] = {}


def _get_morph_patterns(keywords_dict: dict) -> List[Tuple[re.Pattern, str]]:
    import hashlib, json
    key = hashlib.md5(json.dumps(keywords_dict, sort_keys=True).encode()).hexdigest()
    if key not in _PATTERN_CACHE:
        _PATTERN_CACHE[key] = build_dynamic_morph_patterns(keywords_dict)
    return _PATTERN_CACHE[key]


# ── Public API ──────────────────────────────────────────────────────────────────

def normalize_text(text: str, keywords_dict: dict = None) -> str:
    """
    Full normalization pipeline.
    1. Lowercase
    2. Insurance abbreviation expansion (static)
    3. Formatting cleanup — quotes, hyphens, special chars (static)
    4. Morphological normalization — inflected forms → canonical (dynamic,
       built from keywords_dict at runtime)
    5. Final whitespace cleanup
    """
    text = text.lower()

    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    for pattern, replacement in FORMATTING_PATTERNS:
        text = re.sub(pattern, replacement, text)

    if keywords_dict:
        for compiled_pattern, canonical in _get_morph_patterns(keywords_dict):
            text = compiled_pattern.sub(canonical, text)

    return text.strip()


def normalize_keyword(keyword: str, keywords_dict: dict = None) -> str:
    return normalize_text(keyword, keywords_dict)


def normalize_keywords_dict(keywords_dict: dict) -> dict:
    return {
        category: [normalize_keyword(kw, keywords_dict) for kw in keywords]
        for category, keywords in keywords_dict.items()
    }


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def get_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def explain_patterns(keywords_dict: dict) -> None:
    """
    Prints a human-readable report of what variants will be matched
    for each keyword in your dictionary. Call this after loading a new
    keyword file to verify coverage before running analysis.

    Usage:
        from modules.normalizer import explain_patterns
        import json
        explain_patterns(json.load(open('data/keywords/keyword_dictionary.json')))
    """
    print(f"{'='*60}")
    print(f"Dynamic morphological coverage report")
    print(f"Dictionary: {sum(len(v) for v in keywords_dict.values())} keywords "
          f"across {len(keywords_dict)} categories")
    print(f"{'='*60}")
    for category, keywords in keywords_dict.items():
        print(f"\n[{category}]")
        for kw in keywords:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", kw.lower())
            significant = [w for w in words if w not in SKIP_WORDS]
            if not significant:
                print(f"  {kw:35s}  (no significant words to expand)")
                continue
            for word in significant:
                pattern_str, _ = _word_variants_regex(word)
                # Extract readable variants from the pattern
                variants = re.findall(r"(?:\\b|\||\()([a-zA-Z?+*]+)(?:\\b|\)|\|)", pattern_str)
                variants = [v for v in variants if v.isalpha() and len(v) >= 4][:8]
                print(f"  {kw:35s}  [{word}] → {variants}")
