"""
Module: semantic_matcher.py

Semantic similarity matching — upgraded to Azure OpenAI text-embedding-3-small.

Key improvements over the previous sentence-transformers / MiniLM version:

1. Embedding model — text-embedding-3-small (Azure OpenAI) replaces all-MiniLM-L6-v2.
   MiniLM was trained on general English. text-embedding-3-small is a much stronger
   model that understands domain language far better, including insurance terminology,
   legal language, and financial vocabulary.

2. Query construction — instead of embedding raw chunk text and comparing it to
   individual keywords, we now embed CATEGORY-LEVEL queries: a rich description
   of what the category means, with multiple representative phrases. This is why
   semantic search was getting words in wrong context — "fight" against "fight" at
   keyword level is ambiguous, but "fight: prior physical altercations, assault,
   brawl, battery between patrons" is not.

3. Similarity threshold — raised from 0.72 to 0.75 because text-embedding-3-small
   scores are more tightly calibrated. At 0.72 with MiniLM you were getting noise;
   at 0.75 with the new model you get meaningful matches.

4. Context window — matched chunk is expanded to include surrounding sentences
   before being displayed on UI. This fixes the issue where displayed text didn't
   clearly show the matched term.

5. Fallback — if Azure OpenAI is not configured, falls back to sentence-transformers
   automatically so the system still runs locally.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from config.settings import settings
from modules.normalizer import normalize_text


@dataclass
class SemanticMatch:
    category: str
    chunk_index: int
    chunk_text: str           # display text — expanded context window
    chunk_text_raw: str       # original chunk text — used for negation detection
    similarity_score: float
    page: int = None
    section: str = ""
    confidence: str = "Medium"
    match_type: str = "semantic"
    matched_keyword: str = ""


def _dict_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ── Category query builder ─────────────────────────────────────────────────────
# This is the core fix for wrong-context semantic matches.
# Instead of embedding individual keywords ("fight", "fire", "settlement"),
# we embed a rich category-level description that encodes the MEANING
# of the entire category, not just its lexical surface form.

def _build_category_query(category: str, keywords: List[str]) -> str:
    """
    Build a rich query string for a risk category that encodes its meaning
    clearly enough for an embedding model to distinguish contexts.

    The query combines:
    - The category name (the semantic anchor)
    - A plain-English description of what the category means in insurance context
    - Representative keywords from the actual dictionary
    - Context phrases that distinguish this category from superficially similar ones

    Example for "Prior violence losses":
      Bad query (old approach): "assault battery fight shooting"
      Good query (new approach): "Prior violence losses in insurance submission:
        history of physical altercations, assault and battery incidents, fights
        between patrons, shooting incidents, fatality on premises. Indicates
        elevated premises liability and liquor liability risk."
    """
    # Known category descriptions — insurance-domain context
    CATEGORY_DESCRIPTIONS = {
        "Prior violence losses": "history of physical altercations, assaults, battery incidents, fights between patrons, shooting incidents or fatalities on premises — indicates premises liability risk",
        "Prior liquor violations": "previous violations of liquor license terms, fines issued, license suspension or revocation, prior regulatory enforcement actions related to alcohol service",
        "Hazardous processes": "hot work operations, welding, torch cutting, brazing — activities that create ignition sources or fire risk on premises",
        "Special hazard venues": "gun range, firearms range, cannabis dispensary, pyrotechnics storage — high-risk special use premises",
        "High-risk trades": "roofing contractors, demolition work, structural steel erection, tree removal, scaffolding use, crane operations — elevated workers compensation exposure",
        "No safety program": "absence of formal safety manual, no OSHA training, informal safety procedures, no safety certifications held by management",
        "Outdated documentation": "expired permits, missing safety logs, old safety manual not updated, license renewal in process, OSHA violations on record",
        "Poor property condition": "deferred maintenance, poor building condition, lack of lighting, uneven sidewalks, uncovered stairwells — premises liability indicators",
        "Structural & maintenance issues": "foundation settling, bulging or leaning walls, sagging floors, spalling, cracking masonry, water intrusion, leaks, mold, grandfathered systems",
        "Habitational with outdated life-safety systems": "no sprinklers, no fire alarm, no smoke detectors, knob-and-tube wiring, Federal Pacific panels, Stab-Lok panels, aluminum wiring, Zinsco panels",
        "Heavy liquor exposure": "full bar operation, high proof spirits, bottle service, bar-heavy revenue — high liquor sales percentage",
        "Late-night bars / nightclubs": "open after midnight, open after 2am, late hours operation, after hours venue",
        "Aggressive drink promotions": "happy hour promotions, drink specials, all you can drink events, ladies night promotions, dollar beer promotions",
        "Limited liquor controls": "no TIPS certification, no alcohol training program, informal staff training — inadequate responsible service controls",
        "College / nightlife district": "near college campus, college bar, nightlife district — elevated exposure from young patron demographics",
        "New venture bars": "new ownership, newly opened bar, startup bar — limited operating history",
        "BYOB operations": "bring your own bottle policy, BYOB arrangement, corkage fee — customer-supplied alcohol",
        "Live entertainment / dancing": "DJ, live band, dance floor, karaoke, live music — entertainment creating crowd and noise exposure",
        "Amusement devices": "pool tables, arcade games, pinball, shuffleboard, bowling, darts — amusement device liability",
        "High-crime mercantile": "high-crime area location, late-night convenience store, gas station, cash-only business, 24-hour operation",
        "Poor security controls": "untrained door staff, no ID scanner, no surveillance cameras, lax security, bouncers without certification",
        "Labor misclassification": "1099 only workers, cash labor, ghost payroll, misclassified employees — workers compensation exposure",
        "Uninsured subs": "subcontractors without certificates of insurance, no COIs on file, uninsured subcontractors",
        "Weak contracts": "no written contracts, no additional insured endorsement, no hold harmless agreement, verbal-only arrangements",
        "Inconsistent PPE use": "PPE optional policy, inconsistent harness use, inadequate fall protection",
        "Work at heights": "work over 3 stories, exterior work at elevation — fall exposure",
        "Wide radius operations": "multi-state operations, nationwide work, operations beyond 250 miles from home base",
        "Experience gaps": "new to trade, expanding into unfamiliar work types — inexperience indicator",
        "Financial strain": "thin balance sheet, bonding issues, surety concerns — financial distress indicators",
        "Adverse loss history": "frequent falls, struck-by incidents, major property damage claims, worker injuries",
        "Residential new construction": "ground-up construction, custom homes, condos, new homes — new construction liability",
        "Submission inconsistencies": "conflicting information between application and website, inconsistent answers — data quality concerns",
    }

    desc = CATEGORY_DESCRIPTIONS.get(category, "")
    kw_sample = ", ".join(keywords[:8])

    if desc:
        return f"Insurance risk category — {category}: {desc}. Representative terms: {kw_sample}"
    else:
        return f"Insurance submission adverse risk indicator — {category}: {kw_sample}"


# ── Azure OpenAI embedding client ─────────────────────────────────────────────

class _AzureEmbedder:
    """
    Calls Azure OpenAI text-embedding-3-small to produce embeddings.
    Uses the exact pattern from the client's sample code (image 1):
      - AzureOpenAI client with api_version + azure_endpoint + api_key
      - client.embeddings.create(input=[...], model=deployment)
    """

    def __init__(self):
        self._client = None
        self._deployment = settings.azure_openai_embedding_deployment

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
            )
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, returns list of float vectors."""
        client = self._get_client()
        # Batch in groups of 100 (Azure limit per request)
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self._deployment,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class _LocalEmbedder:
    """Fallback: sentence-transformers when Azure OpenAI is not configured."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.warning("Azure OpenAI not configured — falling back to all-MiniLM-L6-v2")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_one(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False).tolist()[0]


def _get_embedder():
    """Return Azure embedder if configured, local fallback otherwise."""
    if settings.azure_openai_endpoint and settings.azure_openai_api_key:
        return _AzureEmbedder()
    return _LocalEmbedder()


# ── Context window expander ────────────────────────────────────────────────────

def _expand_context(chunk_text: str, keyword: str, window_sentences: int = 2) -> str:
    """
    Given a chunk and a matched keyword, return an expanded context string
    that includes the sentence containing the keyword plus surrounding sentences.

    This fixes the UI issue where chunk_text[:500] was showing text that didn't
    contain the matched keyword — because the keyword was found via normalization
    on the full chunk but the truncated display started in the wrong place.
    """
    if not keyword or not chunk_text:
        return chunk_text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
    if len(sentences) <= 1:
        return chunk_text

    # Find which sentence(s) contain the keyword
    kw_lower = keyword.lower()
    kw_sentence_idx = None
    for i, sent in enumerate(sentences):
        if kw_lower in sent.lower():
            kw_sentence_idx = i
            break

    if kw_sentence_idx is None:
        # Keyword not found in any sentence (was matched via normalizer)
        # Return the chunk as-is — already the full paragraph
        return chunk_text

    # Return window_sentences before and after the matching sentence
    start = max(0, kw_sentence_idx - window_sentences)
    end = min(len(sentences), kw_sentence_idx + window_sentences + 1)
    return " ".join(sentences[start:end])


# ── SemanticMatcher ────────────────────────────────────────────────────────────

class SemanticMatcher:

    def __init__(self):
        self._embedder = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        self._current_dict_hash = None
        self._keywords_dict = {}

    def initialize(self, keywords_dict: dict) -> None:
        import chromadb

        dict_hash = _dict_hash(keywords_dict)
        self._keywords_dict = keywords_dict

        # Try Azure embedder first; automatically fall back to local if it fails
        try:
            self._embedder = _get_embedder()
            # Smoke-test the embedder immediately with one short string
            # so we fail fast here rather than silently mid-pipeline
            self._embedder.embed_one("test")
            logger.info(f"Embedder ready: {type(self._embedder).__name__}")
        except Exception as e:
            logger.warning(
                f"Primary embedder ({type(self._embedder).__name__}) "
                f"failed smoke test: {e}\n"
                f"Falling back to local sentence-transformers."
            )
            self._embedder = _LocalEmbedder()

        # Persistent ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=settings.vectordb_path)
        collection_name = "keyword_categories_v2"

        existing_collections = [c.name for c in self.chroma_client.list_collections()]

        rebuild_needed = True
        if collection_name in existing_collections:
            try:
                col = self.chroma_client.get_collection(collection_name)
                meta = col.metadata or {}
                stored_hash    = meta.get("dict_hash", "")
                stored_embedder = meta.get("embedder", "")
                current_embedder = type(self._embedder).__name__

                if (stored_hash == dict_hash
                        and stored_embedder == current_embedder
                        and col.count() > 0):          # must actually have vectors
                    self.collection = col
                    rebuild_needed = False
                    logger.info(
                        f"ChromaDB: reusing collection "
                        f"({col.count()} vectors, hash {dict_hash}, "
                        f"embedder {current_embedder})"
                    )
            except Exception:
                rebuild_needed = True

        if rebuild_needed:
            logger.info(
                f"ChromaDB: rebuilding collection "
                f"(dict hash {dict_hash}, embedder {type(self._embedder).__name__})"
            )
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass

            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dict_hash": dict_hash,
                    "embedder": type(self._embedder).__name__,
                }
            )

            docs, metadatas, ids = [], [], []
            for category, keywords in keywords_dict.items():
                query = _build_category_query(category, keywords)
                docs.append(query)
                metadatas.append({
                    "category": category,
                    "keyword": keywords[0] if keywords else category,
                    "query": query,
                })
                safe_id = (
                    f"cat_{category[:30]}"
                    .replace(" ", "_").replace("/", "_").replace("&", "_")
                )
                ids.append(safe_id)

            if docs:
                logger.info(f"Embedding {len(docs)} category queries...")
                try:
                    embeddings_list = self._embedder.embed(docs)
                    self.collection.add(
                        documents=docs,
                        embeddings=embeddings_list,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    logger.info(
                        f"ChromaDB: stored {len(docs)} category vectors "
                        f"using {type(self._embedder).__name__}"
                    )
                except Exception as e:
                    # Embedding failed even after fallback — log clearly
                    logger.error(
                        f"EMBEDDING FAILED — semantic matching will return 0 results.\n"
                        f"Error: {e}\n"
                        f"Fix: check your Azure OpenAI credentials or ensure "
                        f"sentence-transformers is installed."
                    )
                    self._initialized = False
                    return

        self._initialized = True
        self._current_dict_hash = dict_hash

    def get_collection_stats(self) -> dict:
        if not self._initialized or not self.collection:
            return {}
        return {
            "total_vectors": self.collection.count(),
            "dict_hash": self._current_dict_hash,
            "vectordb_path": settings.vectordb_path,
            "embedder": type(self._embedder).__name__ if self._embedder else "none",
        }

    def match_chunk(self, chunk, top_k: int = 5) -> List[SemanticMatch]:
        """
        Find categories semantically similar to this chunk.

        top_k=5: return up to 5 category matches per chunk.
        Deduplication against exact matches happens in pipeline.py, not here.
        This lets semantic find multiple categories per chunk when relevant.
        """
        if not self._initialized:
            return []

        raw_text = chunk.text
        normalized = normalize_text(raw_text, self._keywords_dict)

        if len(normalized.split()) < 5:
            return []

        try:
            embedding = self._embedder.embed_one(normalized)
        except Exception as e:
            logger.warning(f"Chunk embedding failed (chunk {chunk.chunk_index}): {e}")
            return []

        n = min(top_k, self.collection.count())
        if n == 0:
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances", "documents"],
        )

        matches = []

        for i, distance in enumerate(results["distances"][0]):
            similarity = 1.0 - distance

            if similarity < settings.semantic_similarity_threshold:
                continue

            meta     = results["metadatas"][0][i]
            category = meta["category"]
            keyword  = meta.get("keyword", category)

            # Confidence tier
            if similarity >= 0.85:
                confidence = "High"
            elif similarity >= 0.78:
                confidence = "Medium"
            else:
                confidence = "Low"

            display_text = _expand_context(raw_text, keyword, window_sentences=2)
            if not display_text or len(display_text) < 20:
                display_text = raw_text

            matches.append(SemanticMatch(
                category=category,
                chunk_index=chunk.chunk_index,
                chunk_text=display_text,
                chunk_text_raw=raw_text,
                similarity_score=round(similarity, 3),
                page=chunk.page,
                section=chunk.section or "",
                confidence=confidence,
                match_type="semantic",
                matched_keyword=keyword,
            ))

        return matches

    def match_document(self, ingested_doc) -> List[SemanticMatch]:
        all_matches = []
        for chunk in ingested_doc.chunks:
            all_matches.extend(self.match_chunk(chunk))
        logger.info(
            f"Semantic matching: {len(all_matches)} matches "
            f"in {ingested_doc.filename}"
        )
        return all_matches
