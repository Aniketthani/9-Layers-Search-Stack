"""
Module: llm_interpreter.py

LLM-assisted context interpretation — upgraded to Azure OpenAI gpt-5.4-hamilton.

Uses the exact pattern from the client's sample code (image 2):
  - AzureOpenAI client with api_version + azure_endpoint + api_key
  - client.chat.completions.create(messages=[...], model=deployment)

Provider priority:
  1. azure_openai — if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set
  2. groq           — if GROQ_API_KEY is set
  3. openai         — if OPENAI_API_KEY is set
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


# ── Azure OpenAI ───────────────────────────────────────────────────────────────

def _call_azure_openai(prompt: str) -> dict:
    """
    Calls Azure OpenAI using the pattern from client sample code (image 2):
      from openai import AzureOpenAI
      client = AzureOpenAI(api_version=..., azure_endpoint=..., api_key=...)
      response = client.chat.completions.create(messages=[...], model=deployment)
    """
    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        model=settings.azure_openai_chat_deployment,
        temperature=0.1,
        max_completion_tokens=150,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ── Groq fallback ──────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> dict:
    import httpx
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 150,
    }
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers, json=payload,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)


# ── OpenAI fallback ────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> dict:
    import httpx
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 150,
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers, json=payload,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw)


# ── Provider router ────────────────────────────────────────────────────────────

def _resolve_provider(provider: str) -> str:
    """
    Resolve which provider to actually use.
    azure_openai takes priority if credentials are configured.
    """
    if settings.azure_openai_endpoint and settings.azure_openai_api_key:
        return "azure_openai"
    if provider == "groq" and settings.groq_api_key:
        return "groq"
    if provider == "openai" and settings.openai_api_key:
        return "openai"
    # Auto-detect
    if settings.groq_api_key:
        return "groq"
    if settings.openai_api_key:
        return "openai"
    raise RuntimeError(
        "No LLM provider configured. Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY "
        "in your .env file, or set GROQ_API_KEY / OPENAI_API_KEY as fallback."
    )


def _call_llm(prompt: str, provider: str) -> dict:
    if provider == "azure_openai":
        return _call_azure_openai(prompt)
    elif provider == "groq":
        return _call_groq(prompt)
    else:
        return _call_openai(prompt)


# ── Public API ────────────────────────────────────────────────────────────────

def interpret_match(match, provider: str = None) -> Optional[LLMInterpretation]:
    if match.confidence == "High":
        return None

    resolved_provider = _resolve_provider(provider or settings.llm_provider)

    # Use raw chunk text for LLM interpretation (not the UI-expanded version)
    chunk_text = getattr(match, "chunk_text_raw",
                 getattr(match, "chunk_text", ""))
    category   = getattr(match, "category", "")
    keyword    = getattr(match, "keyword",
                 getattr(match, "matched_keyword", category))

    if not chunk_text or not category:
        return None

    prompt = _build_prompt(chunk_text, category, keyword)

    try:
        result = _call_llm(prompt, resolved_provider)
        return LLMInterpretation(
            chunk_index=match.chunk_index,
            category=category,
            confirmed=result.get("confirmed", False),
            rationale=result.get("rationale", ""),
            provider=resolved_provider,
            confidence_override=result.get("confidence"),
        )
    except Exception as e:
        logger.warning(f"LLM ({resolved_provider}) failed for chunk {match.chunk_index}: {e}")
        return None


def interpret_matches(matches: list, provider: str = None) -> List[LLMInterpretation]:
    resolved = _resolve_provider(provider or settings.llm_provider)
    non_high = [m for m in matches if m.confidence not in ("High", "Informational")]
    logger.info(
        f"LLM interpretation: {len(non_high)} medium/low confidence matches "
        f"via {resolved}"
    )
    results = []
    for match in non_high:
        r = interpret_match(match, provider)
        if r:
            results.append(r)
    return results
