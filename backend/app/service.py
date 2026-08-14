"""Ingestion orchestration: load → chunk → embed → store → refresh BM25."""
from __future__ import annotations

from .config import get_settings
from .ingestion import docmeta
from .ingestion.chunker import chunk_document, new_doc_id
from .ingestion.loader import load_document
from .models import DocumentInfo
from .retrieval import bm25, store
from .retrieval.embeddings import embed_passages
from .tracing_compat import observe


@observe(name="ingest-document")
def ingest_file(file_path: str, doc_name: str, raw_bytes: bytes) -> DocumentInfo:
    """Ingest one document (PDF/DOCX/TXT/MD) and make it queryable."""
    store.ensure_collection()

    doc_id = new_doc_id(doc_name, raw_bytes)
    # Re-uploading the same content replaces the old copy (deterministic id).
    store.delete_document(doc_id)

    pages, source_type = load_document(file_path, doc_name)
    if not pages:
        raise ValueError(
            "No extractable text found in the file (is the PDF scanned images?)."
        )

    chunks = chunk_document(doc_id, doc_name, pages, source_type)
    vectors = embed_passages([c.text for c in chunks])
    store.upsert_chunks(chunks, vectors)
    bm25.invalidate()  # lexical index rebuilds on next query

    # One cheap LLM call for a summary + suggested questions (never fatal).
    meta = docmeta.generate_meta(doc_id, doc_name, chunks) or {}

    return DocumentInfo(
        doc_id=doc_id,
        doc_name=doc_name,
        pages=max(p.page for p in pages),
        chunks=len(chunks),
        source_type=source_type,
        summary=meta.get("summary"),
        suggested_questions=meta.get("questions") or [],
    )


# Backwards-compatible alias (ingest_cli.py, older eval scripts).
ingest_pdf = ingest_file


def remove_document(doc_id: str) -> None:
    store.delete_document(doc_id)
    bm25.invalidate()
    docmeta.delete_meta(doc_id)
    # Remove the stored original file (any extension) alongside the vectors.
    for path in get_settings().upload_dir.glob(f"{doc_id}.*"):
        path.unlink(missing_ok=True)
