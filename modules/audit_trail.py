"""
Module: audit_trail.py
Append-only audit log for every analysis run.
Stores: run metadata, dictionary version, all matches, LLM results, decisions.
Written to JSONL file — one record per line, immutable after write.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from loguru import logger

from config.settings import settings

AUDIT_LOG_PATH = Path(settings.vectordb_path).parent / "audit_trail.jsonl"


def _hash_file(file_path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def _hash_dict(d: dict) -> str:
    serialized = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:16]


def _serialize_match(match) -> dict:
    return {
        "keyword": getattr(match, "keyword", getattr(match, "matched_keyword", "")),
        "category": getattr(match, "category", ""),
        "confidence": getattr(match, "confidence", ""),
        "match_type": getattr(match, "match_type", ""),
        "chunk_index": getattr(match, "chunk_index", -1),
        "page": getattr(match, "page", None),
        "section": getattr(match, "section", ""),
        "affirmed": getattr(match, "affirmed", True),
        "negation_flag": getattr(match, "negation_flag", False),
        "section_weight": getattr(match, "section_weight", 1.0),
        "similarity_score": getattr(match, "similarity_score", None),
    }


def write_audit_record(
    document_path: str,
    keywords_dict: dict,
    all_matches: list,
    llm_interpretations: list,
    cooccurrence_result,
    run_id: str,
    provider: str = "groq",
) -> str:
    record = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "document": {
            "filename": Path(document_path).name,
            "file_hash": _hash_file(document_path),
        },
        "dictionary_version": _hash_dict(keywords_dict),
        "llm_provider": provider,
        "summary": {
            "total_matches": len(all_matches),
            "high_confidence": sum(1 for m in all_matches if m.confidence == "High"),
            "medium_confidence": sum(1 for m in all_matches if m.confidence == "Medium"),
            "low_confidence": sum(1 for m in all_matches if m.confidence == "Low"),
            "negated_matches": sum(1 for m in all_matches if getattr(m, "negation_flag", False)),
            "document_risk_score": cooccurrence_result.document_risk_score if cooccurrence_result else "N/A",
        },
        "matches": [_serialize_match(m) for m in all_matches],
        "llm_interpretations": [
            {
                "chunk_index": i.chunk_index,
                "category": i.category,
                "confirmed": i.confirmed,
                "rationale": i.rationale,
                "provider": i.provider,
            }
            for i in llm_interpretations
        ],
        "cooccurrence": {
            "risk_score": cooccurrence_result.document_risk_score if cooccurrence_result else None,
            "risk_value": cooccurrence_result.risk_score_value if cooccurrence_result else None,
            "triggered_categories": cooccurrence_result.triggered_categories if cooccurrence_result else [],
            "high_risk_combos": [
                {"cat1": c1, "cat2": c2, "weight": w}
                for c1, c2, w in (cooccurrence_result.high_risk_combos_found if cooccurrence_result else [])
            ],
        },
        "underwriter_decisions": [],  # populated later via record_decision()
    }

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(f"Audit record written for run {run_id}")
    return run_id


def record_decision(run_id: str, chunk_index: int, category: str, decision: str, note: str = "") -> None:
    """
    Append an underwriter accept/dismiss decision to the matching audit record.
    Since the file is append-only, decisions are stored as separate delta records.
    """
    delta = {
        "type": "decision",
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "chunk_index": chunk_index,
        "category": category,
        "decision": decision,  # "accept" or "dismiss"
        "note": note,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(delta) + "\n")


def load_audit_records(limit: int = 50) -> list:
    if not AUDIT_LOG_PATH.exists():
        return []
    records = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                continue
    return records[-limit:]
