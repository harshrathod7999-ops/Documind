"""Hybrid retrieval: dense + BM25 → RRF merge → cross-encoder rerank → top-k.

This is DocuMind's differentiator over naive top-k RAG. Three modes are
exposed so the eval harness can quantify the lift from each stage:

  * ``dense``            — baseline: dense vectors only (what most demos ship)
  * ``hybrid``           — dense + BM25 fused with RRF (no rerank)
  * ``hybrid_rerank``    — full pipeline (default): fusion + cross-encoder
"""
from __future__ import annotations

import time
from typing import Literal

from ..config import get_settings
from ..models import RetrievalTrace, RetrievedChunk, TraceEntry
from ..tracing_compat import observe  # thin re-export, keeps imports tidy
from . import bm25, rerank
from .embeddings import embed_query
from .store import dense_search

Mode = Literal["dense", "hybrid", "hybrid_rerank"]


def _rrf_merge(
    ranked_lists: list[list[RetrievedChunk]], rrf_k: int
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank).

    RRF combines rankings without needing the scores to be on the same scale
    (cosine vs BM25 are not comparable), which is exactly why it's the right
    merge for dense + lexical.
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank_pos, item in enumerate(ranked):
            cid = item.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank_pos + 1)
            # keep one representative chunk object per id
            best.setdefault(cid, item)
    fused = [
        RetrievedChunk(chunk=best[cid].chunk, score=score, source="hybrid")
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


@observe(name="retrieve")
def retrieve(
    query: str,
    *,
    mode: Mode = "hybrid_rerank",
    top_k: int | None = None,
    doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    s = get_settings()
    final_k = top_k or s.final_top_k

    if mode == "dense":
        qvec = embed_query(query)
        return dense_search(qvec, final_k, doc_ids)

    # Gather candidates from both retrievers.
    qvec = embed_query(query)
    dense_hits = dense_search(qvec, s.dense_top_k, doc_ids)
    bm25_hits = bm25.bm25_search(query, s.bm25_top_k, doc_ids)
    fused = _rrf_merge([dense_hits, bm25_hits], s.rrf_k)

    if mode == "hybrid":
        return fused[:final_k]

    # Full pipeline: rerank the top fused candidates with the cross-encoder.
    candidates = fused[: s.rerank_candidates]
    return rerank.rerank(query, candidates, final_k)


def _trace_entries(hits: list[RetrievedChunk], limit: int = 10) -> list[TraceEntry]:
    return [
        TraceEntry(
            chunk_id=rc.chunk.id,
            doc_name=rc.chunk.doc_name,
            page=rc.chunk.page,
            rank=i + 1,
            score=round(rc.score, 4),
        )
        for i, rc in enumerate(hits[:limit])
    ]


@observe(name="retrieve-traced")
def retrieve_traced(
    query: str,
    *,
    top_k: int | None = None,
    doc_ids: list[str] | None = None,
) -> tuple[list[RetrievedChunk], RetrievalTrace]:
    """Full hybrid_rerank pipeline, recording each stage for the UI inspector.

    Same behavior as ``retrieve(mode="hybrid_rerank")`` — kept separate so the
    eval harness's ``retrieve()`` stays byte-identical.
    """
    s = get_settings()
    final_k = top_k or s.final_top_k
    timings: dict[str, int] = {}

    t = time.perf_counter()
    qvec = embed_query(query)
    timings["embed"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    dense_hits = dense_search(qvec, s.dense_top_k, doc_ids)
    timings["dense"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    bm25_hits = bm25.bm25_search(query, s.bm25_top_k, doc_ids)
    timings["bm25"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    fused = _rrf_merge([dense_hits, bm25_hits], s.rrf_k)
    timings["fuse"] = int((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    reranked = rerank.rerank(query, fused[: s.rerank_candidates], final_k)
    timings["rerank"] = int((time.perf_counter() - t) * 1000)

    trace = RetrievalTrace(
        query_used=query,
        dense=_trace_entries(dense_hits),
        bm25=_trace_entries(bm25_hits),
        fused=_trace_entries(fused),
        reranked=_trace_entries(reranked, limit=final_k),
        timings_ms=timings,
    )
    return reranked, trace
