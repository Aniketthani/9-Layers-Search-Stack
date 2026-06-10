"""
Module: semantic_matcher.py
Semantic similarity matching using sentence-transformers + ChromaDB (persistent).
ChromaDB collection persists across runs — only rebuilt when dictionary changes.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import List
from loguru import logger

from config.settings import settings
from modules.normalizer import normalize_text


@dataclass
class SemanticMatch:
    category: str
    chunk_index: int
    chunk_text: str
    similarity_score: float
    page: int = None
    section: str = ""
    confidence: str = "Medium"
    match_type: str = "semantic"
    matched_keyword: str = ""


def _dict_hash(d: dict) -> str:
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


class SemanticMatcher:

    def __init__(self):
        self.model = None
        self.chroma_client = None
        self.collection = None
        self._initialized = False
        self._current_dict_hash = None

    def initialize(self, keywords_dict: dict) -> None:
        from sentence_transformers import SentenceTransformer
        import chromadb

        dict_hash = _dict_hash(keywords_dict)

        logger.info("Loading sentence transformer model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Persistent ChromaDB — survives across sessions
        self.chroma_client = chromadb.PersistentClient(path=settings.vectordb_path)

        collection_name = "keyword_categories"
        existing_collections = [c.name for c in self.chroma_client.list_collections()]

        # Check if we need to rebuild (dictionary changed or first run)
        rebuild_needed = True
        if collection_name in existing_collections:
            try:
                col = self.chroma_client.get_collection(collection_name)
                # Store hash in collection metadata to detect dictionary changes
                meta = col.metadata or {}
                if meta.get("dict_hash") == dict_hash:
                    self.collection = col
                    rebuild_needed = False
                    logger.info(f"ChromaDB: reusing existing collection ({col.count()} vectors, hash {dict_hash})")
            except Exception:
                rebuild_needed = True

        if rebuild_needed:
            logger.info(f"ChromaDB: rebuilding collection (dict hash {dict_hash})")
            try:
                self.chroma_client.delete_collection(collection_name)
            except Exception:
                pass

            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "dict_hash": dict_hash}
            )

            docs, metadatas, ids = [], [], []
            for category, keywords in keywords_dict.items():
                for i, kw in enumerate(keywords):
                    norm = normalize_text(kw)
                    docs.append(norm)
                    metadatas.append({"category": category, "keyword": kw})
                    safe_id = f"{category[:20]}_{i}_{kw[:15]}".replace(" ", "_").replace("/", "_")
                    ids.append(safe_id)

            if docs:
                embeddings = self.model.encode(docs, show_progress_bar=False).tolist()
                self.collection.add(documents=docs, embeddings=embeddings,
                                    metadatas=metadatas, ids=ids)
                logger.info(f"ChromaDB: stored {len(docs)} keyword vectors")

        self._initialized = True
        self._current_dict_hash = dict_hash

    def get_collection_stats(self) -> dict:
        if not self._initialized or not self.collection:
            return {}
        return {
            "total_vectors": self.collection.count(),
            "dict_hash": self._current_dict_hash,
            "vectordb_path": settings.vectordb_path,
        }

    def match_chunk(self, chunk, top_k: int = 3) -> List[SemanticMatch]:
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        normalized = normalize_text(chunk.text)
        if len(normalized.split()) < 4:
            return []

        embedding = self.model.encode([normalized], show_progress_bar=False).tolist()[0]
        n = min(top_k, self.collection.count())
        if n == 0:
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["metadatas", "distances", "documents"]
        )

        matches = []
        seen_categories = set()

        for i, distance in enumerate(results["distances"][0]):
            similarity = 1 - (distance / 2)
            if similarity < settings.semantic_similarity_threshold:
                continue

            meta = results["metadatas"][0][i]
            category = meta["category"]
            keyword = meta["keyword"]

            if category in seen_categories:
                continue
            seen_categories.add(category)

            confidence = "Medium" if similarity >= 0.80 else "Low"
            matches.append(SemanticMatch(
                category=category,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
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
        logger.info(f"Semantic matching: {len(all_matches)} matches in {ingested_doc.filename}")
        return all_matches
