"""
pipeline.py — Orchestrator with step-by-step progress callbacks
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from loguru import logger

from config.settings import settings
from modules.document_ingestion import ingest_document
from modules.exact_matcher import ExactMatcher
from modules.semantic_matcher import SemanticMatcher
from modules.negation_detector import apply_negation_detection
from modules.section_weighter import apply_section_weighting
from modules.cooccurrence_scorer import compute_cooccurrence
from modules.llm_interpreter import interpret_matches
from modules.audit_trail import write_audit_record
from modules.documents_db import save_document_result


@dataclass
class StepResult:
    step: int
    name: str
    status: str          # "ok", "warning", "error"
    summary: str
    detail: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    run_id: str
    filename: str
    source_type: str
    all_matches: list
    llm_interpretations: list
    cooccurrence: object
    keywords_dict: dict
    ingested_doc: object
    step_results: List[StepResult] = field(default_factory=list)
    llm_provider: str = "groq"
    errors: List[str] = field(default_factory=list)


class AnalysisPipeline:

    def __init__(self, keywords_dict: dict, llm_provider: str = None):
        self.keywords_dict = keywords_dict
        self.llm_provider = llm_provider or settings.llm_provider

        self.exact_matcher = ExactMatcher()
        self.exact_matcher.build(keywords_dict)

        self.semantic_matcher = SemanticMatcher()
        self.semantic_matcher.initialize(keywords_dict)

        logger.info("Pipeline ready")

    def run(self, file_path: str, run_llm: bool = True,
            progress_cb: Optional[Callable] = None,
            file_bytes: bytes = None) -> PipelineResult:

        run_id = str(uuid.uuid4())[:8]
        errors = []
        step_results = []

        def _step(n, name, fn):
            if progress_cb:
                progress_cb(n, name)
            try:
                result = fn()
                return result, None
            except Exception as e:
                errors.append(f"{name}: {e}")
                logger.warning(f"Step {n} error: {e}")
                return None, str(e)

        # Step 1: Ingest
        if progress_cb: progress_cb(1, "Ingesting document")
        try:
            ingested = ingest_document(file_path)
            step_results.append(StepResult(1, "Document Ingestion", "ok",
                f"{len(ingested.chunks)} chunks extracted from {ingested.source_type.upper()}",
                {"chunks": len(ingested.chunks), "source_type": ingested.source_type,
                 "sections": list(set(c.section for c in ingested.chunks if c.section))}))
        except Exception as e:
            return PipelineResult(run_id=run_id, filename=file_path, source_type="unknown",
                all_matches=[], llm_interpretations=[], cooccurrence=None,
                keywords_dict=self.keywords_dict, ingested_doc=None,
                step_results=[StepResult(1, "Document Ingestion", "error", str(e))],
                errors=[str(e)])

        # Step 2: Exact matching
        if progress_cb: progress_cb(2, "Exact keyword matching (Aho-Corasick)")
        try:
            exact_matches = self.exact_matcher.match_document(ingested)
            cats = list(set(m.category for m in exact_matches))
            step_results.append(StepResult(2, "Exact Matching (Aho-Corasick)", "ok",
                f"{len(exact_matches)} exact matches found across {len(cats)} categories",
                {"match_count": len(exact_matches), "categories_hit": cats,
                 "sample": [{"kw": m.keyword, "cat": m.category, "page": m.page,
                              "section": m.section} for m in exact_matches[:10]]}))
        except Exception as e:
            exact_matches = []
            errors.append(str(e))
            step_results.append(StepResult(2, "Exact Matching", "error", str(e)))

        # Step 3: Semantic matching
        # Semantic runs on ALL chunks independently of exact matching.
        # It looks for categories NOT yet found by exact matching — paraphrased
        # language, synonyms, implied risk. Deduplication happens at results level:
        # if a category was already found by exact matching in the same chunk,
        # the semantic match for that category/chunk is dropped as redundant.
        # But if semantic finds a DIFFERENT category or a DIFFERENT chunk, it is kept.
        if progress_cb: progress_cb(3, "Semantic similarity matching")
        try:
            semantic_matches_raw = self.semantic_matcher.match_document(ingested)

            # Deduplicate: drop semantic matches where exact already found
            # the same (category, chunk_index) — keep all others
            exact_cat_chunks = set(
                (m.category, m.chunk_index) for m in exact_matches
            )
            semantic_matches = [
                m for m in semantic_matches_raw
                if (m.category, m.chunk_index) not in exact_cat_chunks
            ]

            vstats = self.semantic_matcher.get_collection_stats()
            step_results.append(StepResult(3, "Semantic Matching", "ok",
                f"{len(semantic_matches)} semantic matches found "
                f"({len(semantic_matches_raw)} raw, "
                f"{len(semantic_matches_raw) - len(semantic_matches)} deduplicated against exact)",
                {"match_count": len(semantic_matches),
                 "raw_count": len(semantic_matches_raw),
                 "deduped": len(semantic_matches_raw) - len(semantic_matches),
                 "vectordb_vectors": vstats.get("total_vectors", 0),
                 "vectordb_path": vstats.get("vectordb_path", ""),
                 "dict_hash": vstats.get("dict_hash", ""),
                 "embedder": vstats.get("embedder", "unknown"),
                 "sample": [{"cat": m.category, "kw": m.matched_keyword,
                              "score": m.similarity_score} for m in semantic_matches[:8]]}))
        except Exception as e:
            semantic_matches = []
            errors.append(str(e))
            step_results.append(StepResult(3, "Semantic Matching", "error", str(e)))

        all_matches = exact_matches + semantic_matches

        # Step 4: Negation detection
        if progress_cb: progress_cb(4, "Negation detection")
        chunks_map = {c.chunk_index: c.text for c in ingested.chunks}
        all_matches = apply_negation_detection(all_matches, chunks_map)
        negated = sum(1 for m in all_matches if getattr(m, "negation_flag", False))
        step_results.append(StepResult(4, "Negation Detection", "ok",
            f"{negated} matches flagged as negated (denied language)",
            {"negated_count": negated, "affirmed_count": len(all_matches) - negated,
             "negated_examples": [{"kw": getattr(m,"keyword",getattr(m,"matched_keyword","")),
                                   "context": m.chunk_text[:150]}
                                  for m in all_matches if getattr(m,"negation_flag",False)][:3]}))

        # Step 5: Section weighting
        if progress_cb: progress_cb(5, "Section weighting")
        all_matches = apply_section_weighting(all_matches)
        demoted = sum(1 for m in all_matches if m.confidence == "Informational")
        step_results.append(StepResult(5, "Section Weighting", "ok",
            f"{demoted} matches demoted to Informational (found in boilerplate sections)",
            {"demoted_to_informational": demoted,
             "section_breakdown": {s: sum(1 for m in all_matches if getattr(m,"section","") == s)
                                   for s in set(getattr(m,"section","") for m in all_matches)}}))

        # Step 6: Co-occurrence scoring
        if progress_cb: progress_cb(6, "Co-occurrence risk scoring")

        # Safety check: ensure keywords_dict matches the categories in matches
        # If they don't match (stale pipeline), rebuild domain map from triggered cats
        triggered_in_matches = list(set(
            getattr(m, "category", "") for m in all_matches if getattr(m, "category", "")
        ))
        dict_categories = set(self.keywords_dict.keys())
        overlap = len([c for c in triggered_in_matches if c in dict_categories])
        if overlap == 0 and triggered_in_matches:
            # Dict mismatch — build a minimal dict from the triggered categories
            # so domain classification still runs on their names/keywords
            logger.warning(
                f"Dict mismatch: triggered cats not in keywords_dict. "
                f"Triggered: {triggered_in_matches[:3]}. "
                f"Dict has: {list(dict_categories)[:3]}"
            )
            effective_dict = {cat: [cat.lower()] for cat in triggered_in_matches}
        else:
            effective_dict = self.keywords_dict

        cooccurrence = compute_cooccurrence(all_matches, ingested, effective_dict)
        import hashlib as _hl
        _dict_used_hash = _hl.md5(
            json.dumps(effective_dict, sort_keys=True).encode()
        ).hexdigest()[:10]
        step_results.append(StepResult(6, "Co-occurrence Scoring", "ok",
            f"Document risk: {cooccurrence.document_risk_score} — {cooccurrence.notes}",
            {"risk_score": cooccurrence.document_risk_score,
             "risk_value": cooccurrence.risk_score_value,
             "triggered_categories": cooccurrence.triggered_categories,
             "high_risk_combos": [f"{c1} + {c2} (weight {w})"
                                  for c1,c2,w in cooccurrence.high_risk_combos_found],
             "category_frequency": cooccurrence.category_frequency,
             "dict_used_hash": _dict_used_hash,
             "dict_category_count": len(effective_dict),
             "dict_sample_categories": list(effective_dict.keys())[:4]}))

        # Step 7: LLM interpretation
        llm_interpretations = []
        if run_llm:
            if progress_cb: progress_cb(7, f"LLM interpretation ({self.llm_provider})")
            try:
                llm_interpretations = interpret_matches(all_matches, provider=self.llm_provider)
                interp_map = {(i.chunk_index, i.category): i for i in llm_interpretations}
                confirmed = sum(1 for i in llm_interpretations if i.confirmed)
                for match in all_matches:
                    key = (match.chunk_index, getattr(match, "category", ""))
                    if key in interp_map:
                        interp = interp_map[key]
                        match.llm_confirmed = interp.confirmed
                        match.llm_rationale = interp.rationale
                        if interp.confidence_override and match.confidence != "High":
                            match.confidence = interp.confidence_override
                step_results.append(StepResult(7, f"LLM Interpretation ({self.llm_provider})", "ok",
                    f"{len(llm_interpretations)} matches interpreted — {confirmed} confirmed adverse",
                    {"total_interpreted": len(llm_interpretations), "confirmed": confirmed,
                     "denied": len(llm_interpretations) - confirmed,
                     "provider": self.llm_provider,
                     "sample": [{"cat": i.category, "confirmed": i.confirmed,
                                 "rationale": i.rationale} for i in llm_interpretations[:5]]}))
            except Exception as e:
                errors.append(f"LLM: {e}")
                step_results.append(StepResult(7, "LLM Interpretation", "warning",
                    f"LLM unavailable: {e}", {}))
        else:
            for match in all_matches:
                match.llm_confirmed = None
                match.llm_rationale = ""
            step_results.append(StepResult(7, "LLM Interpretation", "warning",
                "Skipped (disabled by user)", {}))

        # Step 8: Audit + Documents DB
        if progress_cb: progress_cb(8, "Saving to audit trail and documents DB")
        try:
            write_audit_record(file_path, self.keywords_dict, all_matches,
                               llm_interpretations, cooccurrence, run_id, self.llm_provider)
            # Save to persistent documents DB
            fb = file_bytes or open(file_path, "rb").read()
            save_document_result(
                run_id=run_id, filename=ingested.filename,
                source_type=ingested.source_type, file_bytes=fb,
                all_matches=all_matches, cooccurrence=cooccurrence,
                keywords_dict=self.keywords_dict, llm_provider=self.llm_provider,
            )
            step_results.append(StepResult(8, "Audit Trail & Documents DB", "ok",
                f"Run {run_id} saved to audit log and documents DB", {"run_id": run_id}))
        except Exception as e:
            errors.append(f"Audit: {e}")
            step_results.append(StepResult(8, "Audit Trail", "warning", str(e)))

        logger.info(f"[{run_id}] Complete — {len(all_matches)} matches, risk: {cooccurrence.document_risk_score}")

        return PipelineResult(
            run_id=run_id, filename=ingested.filename, source_type=ingested.source_type,
            all_matches=all_matches, llm_interpretations=llm_interpretations,
            cooccurrence=cooccurrence, keywords_dict=self.keywords_dict,
            ingested_doc=ingested, step_results=step_results,
            llm_provider=self.llm_provider, errors=errors,
        )


def load_keywords(path: str = None) -> dict:
    path = path or str(settings.keywords_path)
    with open(path, "r") as f:
        return json.load(f)
