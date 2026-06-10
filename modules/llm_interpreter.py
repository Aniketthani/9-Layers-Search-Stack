"""
Module: llm_interpreter.py
LLM-assisted context interpretation for medium/low confidence matches.
Supports Groq and OpenAI. Fixes 'proxies' kwarg issue by using httpx directly.
"""

import json
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from config.settings import settings


@dataclass
class LLMInterpretation:
    chunk_index: int
    category: str
    confirmed: bool
    rationale: str
    provider: str
    confidence_override: Optional[str] = None


SYSTEM_PROMPT = """You are an expert insurance underwriting assistant.
Determine whether a given text excerpt from an insurance submission indicates an adverse risk condition.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{"confirmed": true, "rationale": "one sentence explanation", "confidence": "High"}

confidence must be one of: High, Medium, Low
confirmed must be true or false
Be conservative — only confirm if the text clearly indicates the adverse condition."""


def _build_prompt(chunk_text: str, category: str, keyword: str) -> str:
    return f"""Risk Category: {category}
Matched term: "{keyword}"

Document excerpt:
\"\"\"{chunk_text[:600]}\"\"\"

Does this excerpt indicate an adverse risk condition for "{category}"?"""


def _call_groq(prompt: str) -> dict:
    # Use httpx directly to avoid the 'proxies' kwarg issue with older groq SDK
    import httpx, os
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 150,
    }
    with httpx.Client(timeout=15) as client:
        resp = client.post("https://api.groq.com/openai/v1/chat/completions",
                           headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)


def _call_openai(prompt: str) -> dict:
    import httpx, os
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 150,
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=15) as client:
        resp = client.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=payload)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)


def interpret_match(match, provider: str = None) -> Optional[LLMInterpretation]:
    if match.confidence == "High":
        return None

    provider = provider or settings.llm_provider
    chunk_text = getattr(match, "chunk_text", "")
    category = getattr(match, "category", "")
    keyword = getattr(match, "keyword", getattr(match, "matched_keyword", category))

    if not chunk_text or not category:
        return None

    prompt = _build_prompt(chunk_text, category, keyword)

    try:
        result = _call_groq(prompt) if provider == "groq" else _call_openai(prompt)
        return LLMInterpretation(
            chunk_index=match.chunk_index,
            category=category,
            confirmed=result.get("confirmed", False),
            rationale=result.get("rationale", ""),
            provider=provider,
            confidence_override=result.get("confidence"),
        )
    except Exception as e:
        logger.warning(f"LLM failed for chunk {match.chunk_index}: {e}")
        return None


def interpret_matches(matches: list, provider: str = None) -> List[LLMInterpretation]:
    provider = provider or settings.llm_provider
    non_high = [m for m in matches if m.confidence not in ("High", "Informational")]
    logger.info(f"LLM interpretation: {len(non_high)} medium/low confidence matches via {provider}")
    results = []
    for match in non_high:
        r = interpret_match(match, provider)
        if r:
            results.append(r)
    return results
