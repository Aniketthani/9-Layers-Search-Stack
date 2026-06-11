"""
Module: exact_matcher.py
Aho-Corasick multi-pattern exact matching.

Fix: deduplicate within a chunk — same keyword appearing N times in one
paragraph was producing N identical match cards on the UI.
We keep only the FIRST occurrence per (keyword, category, chunk_index) triplet.
"""

import ahocorasick
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from loguru import logger

from modules.normalizer import normalize_keywords_dict, normalize_text


@dataclass
class ExactMatch:
    keyword: str
    category: str
    start_pos: int
    end_pos: int
    chunk_index: int
    chunk_text: str
    page: int = None
    section: str = ""
    confidence: str = "High"
    match_type: str = "exact"


class ExactMatcher:

    def __init__(self):
        self.automaton = None
        self.keyword_to_category: Dict[str, str] = {}
        self._keywords_dict: dict = {}

    def build(self, keywords_dict: dict) -> None:
        self._keywords_dict = keywords_dict
        normalized = normalize_keywords_dict(keywords_dict)
        self.automaton = ahocorasick.Automaton()
        self.keyword_to_category = {}

        for category, keywords in normalized.items():
            for kw in keywords:
                kw = kw.strip()
                if not kw:
                    continue
                if kw in self.keyword_to_category:
                    existing = self.keyword_to_category[kw]
                    if isinstance(existing, list):
                        existing.append(category)
                    else:
                        self.keyword_to_category[kw] = [existing, category]
                else:
                    self.keyword_to_category[kw] = category
                self.automaton.add_word(kw, kw)

        self.automaton.make_automaton()
        logger.info(
            f"Aho-Corasick built: {len(self.keyword_to_category)} keywords, "
            f"{len(keywords_dict)} categories"
        )

    def match_chunk(self, chunk) -> List[ExactMatch]:
        if self.automaton is None:
            raise RuntimeError("Call build() before match_chunk()")

        normalized_text = normalize_text(chunk.text, self._keywords_dict)
        matches = []

        # ── Deduplication within a chunk ──────────────────────────────────────
        # A keyword may appear multiple times in the same paragraph
        # (e.g. "cannabis dispensary ... cannabis retail licensing").
        # We report it only ONCE per (keyword, category) pair per chunk.
        # The first occurrence is kept because it has the lowest start_pos
        # and is most likely to be the contextually clearest mention.
        seen_in_chunk: Set[Tuple[str, str]] = set()

        for end_pos, keyword in self.automaton.iter(normalized_text):
            start_pos = end_pos - len(keyword) + 1

            # Word boundary check
            before_ok = (start_pos == 0 or
                         not normalized_text[start_pos - 1].isalpha())
            after_ok  = (end_pos + 1 >= len(normalized_text) or
                         not normalized_text[end_pos + 1].isalpha())
            if not (before_ok and after_ok):
                continue

            cat = self.keyword_to_category.get(keyword, "Unknown")
            if isinstance(cat, list):
                cat = cat[0]

            # Skip if we already have this keyword+category in this chunk
            dedup_key = (keyword, cat)
            if dedup_key in seen_in_chunk:
                continue
            seen_in_chunk.add(dedup_key)

            matches.append(ExactMatch(
                keyword=keyword,
                category=cat,
                start_pos=start_pos,
                end_pos=end_pos,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                page=chunk.page,
                section=chunk.section or "",
                confidence="High",
                match_type="exact",
            ))

        return matches

    def match_document(self, ingested_doc) -> List[ExactMatch]:
        all_matches = []
        for chunk in ingested_doc.chunks:
            all_matches.extend(self.match_chunk(chunk))
        logger.info(
            f"Exact matching: {len(all_matches)} matches "
            f"in {ingested_doc.filename}"
        )
        return all_matches
