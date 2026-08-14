"""DocuMind configuration (env-driven, free-tier defaults)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# DocuMind is self-contained: load .env from the project, not a shared repo root.
# config.py lives at 1-documind/backend/app/config.py:
#   parents[1] = 1-documind/backend   parents[2] = 1-documind
_BACKEND_ROOT = Path(__file__).resolve().parents[1]   # 1-documind/backend
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 1-documind


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_PROJECT_ROOT / ".env", _BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Models ---
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    primary_model: str = "groq/openai/gpt-oss-120b"
    fallback_model: str = "gemini/gemini-3.5-flash"
    # Small/fast model for cheap auxiliary calls (follow-up query rewriting,
    # per-document summaries). The router's fallback chain still applies.
    rewrite_model: str = "groq/openai/gpt-oss-20b"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    documind_collection: str = "documind_chunks"

    # --- Chunking ---
    # ~256-token chunks (CLAUDE spec suggested ~400). Smaller chunks gave more
    # precise page citations and finer retrieval granularity on our corpus; for
    # very large documents 400+ trades some precision for a smaller index. See
    # the README "Decisions & trade-offs" section. Tunable via CHUNK_TOKENS.
    chunk_tokens: int = 256
    chunk_overlap_ratio: float = 0.15

    # --- Retrieval ---
    dense_top_k: int = 20               # candidates from dense search
    bm25_top_k: int = 20                # candidates from lexical search
    rrf_k: int = 60                     # RRF damping constant
    final_top_k: int = 5                # after reranking
    rerank_candidates: int = 20         # how many fused candidates to rerank

    # --- API ---
    documind_api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Storage ---
    upload_dir: Path = _BACKEND_ROOT / "data" / "uploads"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def chunk_overlap_tokens(self) -> int:
        return int(self.chunk_tokens * self.chunk_overlap_ratio)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    return s
