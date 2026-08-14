"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3 (local, CPU, free).

Bi-encoder retrieval (dense/BM25) scores query and passage independently; a
cross-encoder reads them *together*, so it's far more precise at ordering the
final few candidates. We over-retrieve (RRF top ~20) then rerank down to top-5.
No paid Cohere reranker — this runs locally.
"""
from __future__ import annotations

import threading
from functools import lru_cache

from ..config import get_settings
from ..models import RetrievedChunk

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model)


def rerank(
    query: str, candidates: list[RetrievedChunk], top_k: int
) -> list[RetrievedChunk]:
    if not candidates:
        return []
    pairs = [[query, c.chunk.text] for c in candidates]
    with _lock:
        scores = _model().predict(pairs, show_progress_bar=False)
    rescored = [
        RetrievedChunk(chunk=c.chunk, score=float(s), source="hybrid")
        for c, s in zip(candidates, scores)
    ]
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored[:top_k]
