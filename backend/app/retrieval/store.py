"""Qdrant dense-vector store + document bookkeeping.

Qdrant holds the dense embeddings and the full chunk payload (text + page +
section), so it doubles as our source of truth for the BM25 index, which is
rebuilt in-memory from the payloads (see ``bm25.py``).
"""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import get_settings
from ..models import Chunk, DocumentInfo, RetrievedChunk
from .embeddings import embedding_dim


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)


def _doc_filter(doc_ids: list[str] | None) -> qm.Filter | None:
    if not doc_ids:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))]
    )


def ensure_collection() -> None:
    client = get_client()
    name = get_settings().documind_collection
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=embedding_dim(), distance=qm.Distance.COSINE
        ),
    )
    # Index doc_id so per-document filtering stays fast.
    client.create_payload_index(
        collection_name=name,
        field_name="doc_id",
        field_schema=qm.PayloadSchemaType.KEYWORD,
    )


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    client = get_client()
    name = get_settings().documind_collection
    points = [
        qm.PointStruct(id=c.id, vector=v, payload=c.model_dump())
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=name, points=points, wait=True)


def _payload_to_chunk(payload: dict) -> Chunk:
    return Chunk(**payload)


def dense_search(
    query_vector: list[float], top_k: int, doc_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    client = get_client()
    name = get_settings().documind_collection
    hits = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        query_filter=_doc_filter(doc_ids),
        with_payload=True,
    ).points
    return [
        RetrievedChunk(
            chunk=_payload_to_chunk(h.payload), score=float(h.score), source="dense"
        )
        for h in hits
    ]


def scroll_all_chunks(doc_ids: list[str] | None = None) -> list[Chunk]:
    """Fetch every stored chunk (optionally filtered) — used to build BM25."""
    client = get_client()
    name = get_settings().documind_collection
    out: list[Chunk] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=name,
            scroll_filter=_doc_filter(doc_ids),
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        out.extend(_payload_to_chunk(r.payload) for r in records)
        if offset is None:
            break
    return out


def list_documents() -> list[DocumentInfo]:
    docs: dict[str, DocumentInfo] = {}
    for c in scroll_all_chunks():
        info = docs.get(c.doc_id)
        if info is None:
            docs[c.doc_id] = DocumentInfo(
                doc_id=c.doc_id,
                doc_name=c.doc_name,
                pages=c.page,
                chunks=1,
                source_type=c.source_type,
            )
        else:
            info.chunks += 1
            info.pages = max(info.pages, c.page)
    return sorted(docs.values(), key=lambda d: d.doc_name)


def delete_document(doc_id: str) -> None:
    client = get_client()
    name = get_settings().documind_collection
    client.delete(
        collection_name=name,
        points_selector=qm.FilterSelector(filter=_doc_filter([doc_id])),
        wait=True,
    )


def collection_count() -> int:
    client = get_client()
    name = get_settings().documind_collection
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name).count
