"""
Module: documents_db.py
Persistent storage of every analyzed document and its results.
Stored as JSONL — one record per document analysis.
Supports listing all documents, retrieving by run_id, and search.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from config.settings import settings

DB_PATH = Path(settings.docs_db_path)
DB_FILE = DB_PATH / "analyzed_documents.jsonl"


def _ensure_dir():
    DB_PATH.mkdir(parents=True, exist_ok=True)


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def save_document_result(
    run_id: str,
    filename: str,
    source_type: str,
    file_bytes: bytes,
    all_matches: list,
    cooccurrence,
    keywords_dict: dict,
    llm_provider: str,
) -> None:
    _ensure_dir()

    def _ser(m):
        return {
            "keyword":     getattr(m, "keyword", getattr(m, "matched_keyword", "")),
            "category":    getattr(m, "category", ""),
            "confidence":  getattr(m, "confidence", ""),
            "match_type":  getattr(m, "match_type", ""),
            "page":        getattr(m, "page", None),
            "section":     getattr(m, "section", ""),
            "affirmed":    getattr(m, "affirmed", True),
            "negation_flag": getattr(m, "negation_flag", False),
            "similarity_score": getattr(m, "similarity_score", None),
            "llm_confirmed":  getattr(m, "llm_confirmed", None),
            "llm_rationale":  getattr(m, "llm_rationale", ""),
            "chunk_text":  getattr(m, "chunk_text", "")[:400],
        }

    record = {
        "run_id":       run_id,
        "timestamp":    datetime.utcnow().isoformat(),
        "filename":     filename,
        "source_type":  source_type,
        "file_hash":    _file_hash(file_bytes),
        "file_size_kb": round(len(file_bytes) / 1024, 1),
        "llm_provider": llm_provider,
        "dict_version": hashlib.md5(json.dumps(keywords_dict, sort_keys=True).encode()).hexdigest()[:10],
        "risk_score":   cooccurrence.document_risk_score if cooccurrence else "N/A",
        "risk_value":   cooccurrence.risk_score_value if cooccurrence else 0,
        "triggered_categories": cooccurrence.triggered_categories if cooccurrence else [],
        "high_risk_combos": [
            {"cat1": c1, "cat2": c2, "weight": w}
            for c1, c2, w in (cooccurrence.high_risk_combos_found if cooccurrence else [])
        ],
        "category_frequency": cooccurrence.category_frequency if cooccurrence else {},
        "summary": {
            "total_matches":    len(all_matches),
            "high_confidence":  sum(1 for m in all_matches if m.confidence == "High"),
            "medium_confidence":sum(1 for m in all_matches if m.confidence == "Medium"),
            "low_confidence":   sum(1 for m in all_matches if m.confidence == "Low"),
            "negated":          sum(1 for m in all_matches if getattr(m,"negation_flag",False)),
            "exact_matches":    sum(1 for m in all_matches if getattr(m,"match_type","")=="exact"),
            "semantic_matches": sum(1 for m in all_matches if getattr(m,"match_type","")=="semantic"),
        },
        "matches": [_ser(m) for m in all_matches],
    }

    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_all_documents(limit: int = 100) -> List[Dict]:
    _ensure_dir()
    if not DB_FILE.exists():
        return []
    records = []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                continue
    return list(reversed(records[-limit:]))


def get_document_by_run_id(run_id: str) -> Optional[Dict]:
    for rec in load_all_documents(500):
        if rec.get("run_id") == run_id:
            return rec
    return None


def get_db_stats() -> Dict:
    records = load_all_documents(1000)
    if not records:
        return {"total": 0}
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    for r in records:
        risk = r.get("risk_score", "")
        if risk in risk_counts:
            risk_counts[risk] += 1
    return {
        "total": len(records),
        "high_risk": risk_counts["High"],
        "medium_risk": risk_counts["Medium"],
        "low_risk": risk_counts["Low"],
        "unique_files": len(set(r.get("filename","") for r in records)),
    }
