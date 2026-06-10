from pydantic import BaseModel
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str   = os.getenv("GROQ_API_KEY", "")
    openai_model: str   = "gpt-4o-mini"
    groq_model: str     = "llama-3.3-70b-versatile"   # fixed — llama3-70b-8192 is deprecated
    llm_provider: str   = "groq"

    keywords_path: Path  = BASE_DIR / "data" / "keywords" / "business_operation_keywords.json"
    vectordb_path: str   = str(BASE_DIR / "data" / "vectordb")
    docs_db_path: str    = str(BASE_DIR / "data" / "documents_db")
    sample_docs_path: Path = BASE_DIR / "data" / "sample_documents"

    semantic_similarity_threshold: float = 0.72
    negation_window_tokens: int          = 6
    cooccurrence_window_paragraphs: int  = 3

    high_confidence_label: str   = "High"
    medium_confidence_label: str = "Medium"
    low_confidence_label: str    = "Low"

settings = Settings()
