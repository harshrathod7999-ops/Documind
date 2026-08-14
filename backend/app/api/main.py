"""DocuMind FastAPI app: ingest, list/delete docs, ask (sync + SSE stream)."""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from ..config import get_settings
from ..generation.answer import _collect_citations, generate_answer, stream_answer
from ..generation.rewrite import condense_question
from ..ingestion import docmeta
from ..ingestion.chunker import new_doc_id
from ..ingestion.loader import SUPPORTED_EXTENSIONS
from ..models import (
    AnswerResponse,
    AskRequest,
    DocumentInfo,
    IngestResponse,
)
from ..retrieval import store
from ..retrieval.pipeline import retrieve, retrieve_traced
from ..service import ingest_file, remove_document
from ..tracing_compat import flush
from ..warmup import is_ready, start_warmup

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the embedding + reranker models in the background so the first
    # user query doesn't pay the cold-load cost. /health exposes readiness.
    start_warmup()
    yield


app = FastAPI(
    title="DocuMind API",
    description="RAG document intelligence with hybrid retrieval + reranking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        count = store.collection_count()
        qdrant_ok = True
    except Exception:
        count, qdrant_ok = 0, False
    return {
        "status": "ok",
        "qdrant_ok": qdrant_ok,
        "indexed_chunks": count,
        # False during the one-time model warm-up after server start.
        "models_ready": is_ready(),
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(400, "No files uploaded.")
    docs: list[DocumentInfo] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                400,
                f"{f.filename}: unsupported file type. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
            )
        # Strip any directory components to prevent path traversal (../../x.pdf).
        safe_name = Path(f.filename).name
        raw = await f.read()
        if not raw:
            raise HTTPException(400, f"{safe_name}: empty file.")
        # Store under the deterministic doc_id so same-named files can't
        # overwrite each other's bytes, and citations can serve the original.
        doc_id = new_doc_id(safe_name, raw)
        dest = settings.upload_dir / f"{doc_id}{ext}"
        dest.write_bytes(raw)
        try:
            info = ingest_file(str(dest), safe_name, raw)
        except ValueError as e:
            dest.unlink(missing_ok=True)
            detail: dict | str = str(e)
            if "scanned" in str(e).lower():
                detail = {
                    "code": "scanned_pdf",
                    "message": f"{safe_name}: this PDF appears to be scanned images "
                    "without extractable text. OCR isn't supported — export a "
                    "text-based PDF instead.",
                }
            raise HTTPException(422, detail)
        docs.append(info)
    return IngestResponse(documents=docs, total_chunks=sum(d.chunks for d in docs))


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    docs = store.list_documents()
    # Merge sidecar intelligence (summary + suggested questions) if present.
    for d in docs:
        meta = docmeta.read_meta(d.doc_id)
        if meta:
            d.summary = meta.get("summary")
            d.suggested_questions = meta.get("questions") or []
    return docs


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    remove_document(doc_id)
    return {"deleted": doc_id}


_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


@app.get("/documents/{doc_id}/file")
def document_file(doc_id: str) -> FileResponse:
    """Serve the original uploaded file (used by the citation PDF viewer)."""
    doc = next((d for d in store.list_documents() if d.doc_id == doc_id), None)
    if doc is None:
        raise HTTPException(404, "Unknown document.")
    upload_dir = settings.upload_dir.resolve()
    candidates = [upload_dir / f"{doc_id}{ext}" for ext in _MEDIA_TYPES]
    # Legacy fallback: files stored under their original name before the
    # doc_id-based storage scheme.
    candidates.append(upload_dir / Path(doc.doc_name).name)
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file() and resolved.parent == upload_dir:
            return FileResponse(
                resolved,
                media_type=_MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream"),
                filename=doc.doc_name,
                content_disposition_type="inline",
            )
    raise HTTPException(404, "Original file is no longer available on the server.")


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> AnswerResponse:
    t0 = time.perf_counter()
    # Follow-ups are condensed into a standalone query before retrieval;
    # first turns (no history) skip the extra LLM call entirely.
    search_query, rewritten = condense_question(req.question, req.history)
    if req.include_trace:
        chunks, trace = retrieve_traced(
            search_query, top_k=req.top_k, doc_ids=req.doc_ids
        )
        trace.rewritten = rewritten
    else:
        chunks = retrieve(
            search_query, mode="hybrid_rerank", top_k=req.top_k, doc_ids=req.doc_ids
        )
        trace = None
    t1 = time.perf_counter()
    resp = generate_answer(req.question, chunks, req.history)
    t2 = time.perf_counter()
    resp.retrieval_ms = int((t1 - t0) * 1000)
    resp.generation_ms = int((t2 - t1) * 1000)
    resp.trace = trace
    flush()
    return resp


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Server-Sent Events: meta + sources (+ trace) first, then answer tokens."""
    search_query, rewritten = condense_question(req.question, req.history)
    if req.include_trace:
        chunks, trace = retrieve_traced(
            search_query, top_k=req.top_k, doc_ids=req.doc_ids
        )
        trace.rewritten = rewritten
    else:
        chunks = retrieve(
            search_query, mode="hybrid_rerank", top_k=req.top_k, doc_ids=req.doc_ids
        )
        trace = None

    def event_gen():
        # Tell the UI what was actually searched (follow-up rewriting).
        meta = {"query_used": search_query, "rewritten": rewritten}
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        # Emit sources up-front so the UI can render citation chips immediately.
        sources = [
            {
                "marker": i + 1,
                "chunk_id": rc.chunk.id,
                "doc_id": rc.chunk.doc_id,
                "doc_name": rc.chunk.doc_name,
                "page": rc.chunk.page,
                "section": rc.chunk.section,
                "score": rc.score,
            }
            for i, rc in enumerate(chunks)
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

        if trace is not None:
            yield f"event: trace\ndata: {trace.model_dump_json()}\n\n"

        buffer = []
        for token in stream_answer(req.question, chunks, req.history):
            buffer.append(token)
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        answer = "".join(buffer)
        citations, grounded = _collect_citations(answer, chunks)
        payload = {
            "grounded": grounded,
            "citations": [c.model_dump() for c in citations],
        }
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"
        flush()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def root() -> dict:
    return {"name": "DocuMind", "docs": "/docs", "health": "/health"}
