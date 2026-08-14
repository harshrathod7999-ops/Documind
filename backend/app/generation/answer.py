"""Grounded answer synthesis with page-level citations.

Contract enforced on the model:
  * Use ONLY the numbered context. If the answer isn't there, say so plainly
    ("I couldn't find this in the documents.") — no hallucinated filler.
  * Cite with inline [n] markers that map to the provided sources.
  * When sources disagree, surface the conflict rather than silently picking one.

We then post-process: only markers the model actually used become citations,
so the UI never shows a dangling source.
"""
from __future__ import annotations

import re

from ..llm_compat import chat, chat_stream
from ..models import AnswerResponse, ChatTurn, Citation, RetrievedChunk

_NOT_FOUND = "I couldn't find this in the documents."

SYSTEM_PROMPT = """You are DocuMind, a precise document-grounded assistant.

Rules:
1. Answer ONLY from the numbered SOURCES below. Never use outside knowledge.
2. If the sources do not contain the answer, reply exactly: "{not_found}"
3. Cite every claim with inline markers like [1] or [2][3] referring to the \
SOURCE numbers you used. Place the marker right after the sentence it supports.
4. If sources conflict, state the conflict explicitly and cite each side.
5. Be concise and factual. Do not pad. Do not repeat the question.
6. Format with Markdown (short lists, tables, bold) when it aids clarity; \
keep simple answers as plain prose.
""".format(not_found=_NOT_FOUND)

_MARKER_RE = re.compile(r"\[(\d+)(?:[^\]]*)\]")

# Some models (notably gpt-oss) emit citations with fullwidth CJK brackets
# (【1】) despite the [n] instruction. Normalize so markers stay clickable.
_BRACKET_FIX = str.maketrans({"【": "[", "】": "]"})


def _snippet(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered SOURCES block."""
    lines = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        loc = f"{c.doc_name}, page {c.page}"
        if c.section:
            loc += f", section “{c.section}”"
        lines.append(f"[{i}] ({loc})\n{c.text}")
    return "\n\n".join(lines)


def _build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ChatTurn] | None = None,
) -> list[dict]:
    context = build_context(chunks)
    user = (
        f"SOURCES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer with inline [n] citations following the rules."
    )
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Prior turns give the model conversational context (e.g. what "it" means);
    # grounding still comes only from the fresh SOURCES in the final message.
    for turn in (history or [])[-6:]:
        messages.append({"role": turn.role, "content": turn.content[:2000]})
    messages.append({"role": "user", "content": user})
    return messages


def _collect_citations(
    answer: str, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], bool]:
    used = {int(m) for m in _MARKER_RE.findall(answer)}
    grounded = bool(used) and _NOT_FOUND.lower() not in answer.lower()
    citations: list[Citation] = []
    for marker in sorted(used):
        if 1 <= marker <= len(chunks):
            rc = chunks[marker - 1]
            c = rc.chunk
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=c.doc_id,
                    doc_name=c.doc_name,
                    page=c.page,
                    section=c.section,
                    snippet=_snippet(c.text),
                    score=rc.score,
                    source_type=c.source_type,
                )
            )
    return citations, grounded


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ChatTurn] | None = None,
) -> AnswerResponse:
    if not chunks:
        return AnswerResponse(answer=_NOT_FOUND, citations=[], grounded=False)
    messages = _build_messages(question, chunks, history)
    answer = chat(
        messages,
        temperature=0.1,
        max_tokens=900,
        metadata={"trace_name": "documind-answer"},
    ).strip().translate(_BRACKET_FIX)
    citations, grounded = _collect_citations(answer, chunks)
    return AnswerResponse(answer=answer, citations=citations, grounded=grounded)


def stream_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ChatTurn] | None = None,
):
    """Yield answer tokens for SSE. Citations are computed by the caller after."""
    if not chunks:
        yield _NOT_FOUND
        return
    messages = _build_messages(question, chunks, history)
    for token in chat_stream(
        messages, temperature=0.1, max_tokens=900,
        metadata={"trace_name": "documind-answer-stream"},
    ):
        # Safe per-token: each bracket is a single character, so the mapping
        # holds even when markers split across token boundaries.
        yield token.translate(_BRACKET_FIX)
