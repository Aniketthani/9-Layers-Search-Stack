from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    # ── Legacy providers (kept for fallback) ──────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key:   str = os.getenv("GROQ_API_KEY", "")
    openai_model:   str = "gpt-4o-mini"
    groq_model:     str = "llama-3.3-70b-versatile"
    llm_provider:   str = "groq"

    # ── Azure OpenAI ──────────────────────────────────────────────────────
    # Endpoint from image: https://aifoundryjun2026.cognitiveservices.azure.com/
    azure_openai_endpoint:    str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key:     str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # LLM deployment — gpt-5.4-hamilton (from image 2)
    azure_openai_chat_deployment:  str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.4-hamilton")
    azure_openai_chat_model:       str = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-5.4")

    # Embedding deployment — text-embedding-3-small (from image 1)
    azure_openai_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    azure_openai_embedding_model:      str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # text-embedding-3-small produces 1536-dim vectors
    azure_openai_embedding_dimensions: int = 1536

    # ── Paths ─────────────────────────────────────────────────────────────
    keywords_path:    Path = BASE_DIR / "data" / "keywords" / "business_operation_keywords.json"
    vectordb_path:    str  = str(BASE_DIR / "data" / "vectordb")
    docs_db_path:     str  = str(BASE_DIR / "data" / "documents_db")
    sample_docs_path: Path = BASE_DIR / "data" / "sample_documents"

    # ── Matching config ───────────────────────────────────────────────────
    # Raised from 0.72 — text-embedding-3-small scores differently to MiniLM
    # 0.75 gives better precision with the new model
    semantic_similarity_threshold:    float = 0.75
    negation_window_tokens:           int   = 6
    cooccurrence_window_paragraphs:   int   = 3

    high_confidence_label:   str = "High"
    medium_confidence_label: str = "Medium"
    low_confidence_label:    str = "Low"

settings = Settings()
