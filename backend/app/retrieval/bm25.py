"""In-memory BM25 lexical retriever, rebuilt from Qdrant payloads.

BM25 catches the queries dense vectors are worst at: exact IDs, codes,
SKUs, error numbers, rare proper nouns — the literal-match cases. Pairing
it with dense search (and fusing via RRF) is what lifts recall over naive
top-k. The index is small (chunk count), so rebuilding on ingest is cheap.
"""
from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from ..models import Chunk, RetrievedChunk
from . import store

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _Bm25Cache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._dirty = True

    def invalidate(self) -> None:
        with self._lock:
            self._dirty = True

    def _rebuild_locked(self) -> None:
        self._chunks = store.scroll_all_chunks()
        corpus = [_tokenize(c.text) for c in self._chunks]
        # BM25Okapi needs a non-empty corpus; guard the empty case.
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._dirty = False

    def search(
        self, query: str, top_k: int, doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        with self._lock:
            if self._dirty or self._bm25 is None:
                self._rebuild_locked()
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(_tokenize(query))
            chunks = self._chunks

        ranked = sorted(
            zip(chunks, scores), key=lambda cs: cs[1], reverse=True
        )
        results: list[RetrievedChunk] = []
        allowed = set(doc_ids) if doc_ids else None
        for chunk, score in ranked:
            if score <= 0:
                continue
            if allowed and chunk.doc_id not in allowed:
                continue
            results.append(
                RetrievedChunk(chunk=chunk, score=float(score), source="bm25")
            )
            if len(results) >= top_k:
                break
        return results


_cache = _Bm25Cache()


def invalidate() -> None:
    _cache.invalidate()


def bm25_search(
    query: str, top_k: int, doc_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    return _cache.search(query, top_k, doc_ids)
