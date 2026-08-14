"""Pydantic v2 schemas for DocuMind structured I/O."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single retrievable unit, tied back to its source page."""

    id: str = Field(..., description="Deterministic uuid5 over doc_id+chunk_index.")
    doc_id: str
    doc_name: str
    page: int = Field(..., description="1-based page (PDF) or pseudo-page part number.")
    chunk_index: int
    text: str
    section: str | None = Field(None, description="Nearest heading, if detected.")
    token_count: int = 0
    source_type: str = Field("pdf", description="pdf | docx | text")


class Citation(BaseModel):
    """Page-level source citation surfaced to the UI."""

    marker: int = Field(..., description="The [n] number used inline in the answer.")
    doc_id: str
    doc_name: str
    page: int
    section: str | None = None
    snippet: str = Field(..., description="Short supporting excerpt from the chunk.")
    score: float = Field(0.0, description="Reranker relevance score.")
    source_type: str = Field("pdf", description="Gates the PDF viewer in the UI.")


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: str = Field("hybrid", description="dense | bm25 | hybrid (post-fusion).")


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    pages: int
    chunks: int
    source_type: str = "pdf"
    summary: str | None = None
    suggested_questions: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class ChatTurn(BaseModel):
    """One prior turn of the conversation, for follow-up question rewriting."""

    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: list[str] | None = Field(
        None, description="Restrict retrieval to these documents; None = all."
    )
    top_k: int | None = Field(None, ge=1, le=20)
    history: list[ChatTurn] | None = Field(
        None,
        max_length=12,
        description="Prior turns; follow-ups are condensed into a standalone "
        "search query before retrieval.",
    )
    include_trace: bool = Field(
        False, description="Emit per-stage retrieval trace (dense/BM25/RRF/rerank)."
    )


class TraceEntry(BaseModel):
    """One candidate chunk at one stage of the retrieval pipeline."""

    chunk_id: str
    doc_name: str
    page: int
    rank: int
    score: float = Field(..., description="Stage-native score (cosine | bm25 | rrf | rerank).")


class RetrievalTrace(BaseModel):
    """Per-stage retrieval transparency for the UI's 'how was this found' panel."""

    query_used: str
    rewritten: bool = False
    dense: list[TraceEntry]
    bm25: list[TraceEntry]
    fused: list[TraceEntry]
    reranked: list[TraceEntry]
    timings_ms: dict[str, int] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool = Field(
        True, description="False when the model fell back to 'not in the documents'."
    )
    retrieval_ms: int = 0
    generation_ms: int = 0
    trace: RetrievalTrace | None = None
