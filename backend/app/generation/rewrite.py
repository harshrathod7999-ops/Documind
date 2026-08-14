"""Condense a follow-up question + chat history into a standalone search query.

Follow-ups like "what about the X300?" retrieve poorly as-is: the vectors and
BM25 terms live in the *previous* turns. One cheap LLM call rewrites the
follow-up into a self-contained query before retrieval. First turns skip this
entirely, and any failure falls back to the raw question — the rewrite must
never break /ask.
"""
from __future__ import annotations

import re

from ..config import get_settings
from ..llm_compat import chat
from ..models import ChatTurn
from ..tracing_compat import observe

_MARKER_RE = re.compile(r"\[\d+\]")

_SYSTEM = (
    "You rewrite a follow-up question from a document-Q&A conversation into "
    "ONE standalone search query. Resolve pronouns and references using the "
    "conversation. Keep exact identifiers, error codes, and product names "
    "verbatim. If the question is already self-contained, return it unchanged. "
    "Return ONLY the query text — no quotes, no explanation."
)


@observe(name="condense-question")
def condense_question(
    question: str, history: list[ChatTurn] | None
) -> tuple[str, bool]:
    """Return (search_query, was_rewritten). No-op on first turns."""
    if not history:
        return question, False

    convo = []
    for turn in history[-6:]:
        text = _MARKER_RE.sub("", turn.content).strip()
        convo.append(f"{turn.role.upper()}: {text[:500]}")
    user = (
        "CONVERSATION:\n" + "\n".join(convo) + f"\n\nFOLLOW-UP QUESTION: {question}"
    )

    try:
        rewritten = chat(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            model=get_settings().rewrite_model,
            temperature=0.0,
            # gpt-oss is a reasoning model: it spends tokens thinking before
            # emitting content, so a tight cap yields an EMPTY completion.
            max_tokens=600,
            metadata={"trace_name": "documind-condense"},
        ).strip().strip('"')
    except Exception:
        return question, False

    # Guard against degenerate rewrites (empty, or a runaway explanation).
    if not rewritten or len(rewritten) > 400 or "\n" in rewritten:
        return question, False
    return rewritten, rewritten.lower() != question.lower()
