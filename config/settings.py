from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Streamlit Cloud secrets fallback ──────────────────────────────────────────
# On Streamlit Cloud, secrets are in st.secrets (not env vars).
# Locally, secrets come from .env via load_dotenv() above.
def _get(key: str, default: str = "") -> str:
    """Read from env first, then Streamlit secrets, then default."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


class Settings(BaseModel):
    # ── Legacy providers ──────────────────────────────────────────────────
    openai_api_key: str = ""
    groq_api_key:   str = ""
    openai_model:   str = "gpt-4o-mini"
    groq_model:     str = "llama-3.3-70b-versatile"
    llm_provider:   str = "groq"

    # ── Azure OpenAI ──────────────────────────────────────────────────────
    azure_openai_endpoint:             str = ""
    azure_openai_api_key:              str = ""
    azure_openai_api_version:          str = "2024-12-01-preview"
    azure_openai_chat_deployment:      str = "gpt-5.4-hamilton"
    azure_openai_chat_model:           str = "gpt-5.4"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_embedding_model:      str = "text-embedding-3-small"
    azure_openai_embedding_dimensions: int = 1536

    # ── Paths ─────────────────────────────────────────────────────────────
    keywords_path:    Path = BASE_DIR / "data" / "keywords" / "business_operation_keywords.json"
    vectordb_path:    str  = str(BASE_DIR / "data" / "vectordb")
    docs_db_path:     str  = str(BASE_DIR / "data" / "documents_db")
    sample_docs_path: Path = BASE_DIR / "data" / "sample_documents"

    # ── Matching config ───────────────────────────────────────────────────
    semantic_similarity_threshold:  float = 0.80
    negation_window_tokens:         int   = 6
    cooccurrence_window_paragraphs: int   = 3
    high_confidence_label:          str   = "High"
    medium_confidence_label:        str   = "Medium"
    low_confidence_label:           str   = "Low"


def _build_settings() -> Settings:
    return Settings(
        openai_api_key             = _get("OPENAI_API_KEY"),
        groq_api_key               = _get("GROQ_API_KEY"),
        azure_openai_endpoint      = _get("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key       = _get("AZURE_OPENAI_API_KEY"),
        azure_openai_api_version   = _get("AZURE_OPENAI_API_VERSION",
                                          "2024-12-01-preview"),
        azure_openai_chat_deployment      = _get("AZURE_OPENAI_CHAT_DEPLOYMENT",
                                                  "gpt-5.4-hamilton"),
        azure_openai_chat_model           = _get("AZURE_OPENAI_CHAT_MODEL", "gpt-5.4"),
        azure_openai_embedding_deployment = _get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                                                  "text-embedding-3-small"),
        azure_openai_embedding_model      = _get("AZURE_OPENAI_EMBEDDING_MODEL",
                                                  "text-embedding-3-small"),
    )


settings = _build_settings()
