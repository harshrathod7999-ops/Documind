"""Per-document intelligence: a short summary + suggested questions.

Generated once at ingest with one cheap LLM call over the first few chunks and
stored as a sidecar JSON at data/meta/{doc_id}.json — NOT in Qdrant, where a
summary duplicated onto every chunk point would bloat payloads and a synthetic
"meta point" would pollute search. Any failure degrades to no summary; it must
never fail an ingest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import get_settings
from ..llm_compat import chat
from ..models import Chunk
from ..tracing_compat import observe


def _meta_dir() -> Path:
    d = get_settings().upload_dir.parent / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_meta(doc_id: str) -> dict | None:
    path = _meta_dir() / f"{doc_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_meta(doc_id: str) -> None:
    (_meta_dir() / f"{doc_id}.json").unlink(missing_ok=True)


_PROMPT = (
    "You index documents for a Q&A system. Given the beginning of a document, "
    "return STRICT JSON (no prose, no code fences) with exactly these keys:\n"
    '{"summary": "<what this document is, in at most 40 words>", '
    '"questions": ["<3 short, specific questions this document can answer>"]}'
)


@observe(name="doc-meta")
def generate_meta(doc_id: str, doc_name: str, chunks: list[Chunk]) -> dict | None:
    """Summarize the doc + propose questions. Returns the meta dict or None."""
    excerpt = "\n\n".join(c.text for c in chunks[:6])[:6000]
    if not excerpt.strip():
        return None
    try:
        raw = chat(
            [
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": f"DOCUMENT: {doc_name}\n\n{excerpt}"},
            ],
            model=get_settings().rewrite_model,
            temperature=0.2,
            # Generous cap: gpt-oss reasons before answering (see rewrite.py).
            max_tokens=800,
            metadata={"trace_name": "documind-docmeta"},
        )
        # Parse leniently: take the first {...} block (some models add chatter).
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0) if m else raw)
        meta = {
            "summary": str(data.get("summary", ""))[:400] or None,
            "questions": [str(q)[:200] for q in (data.get("questions") or [])][:3],
        }
    except Exception:
        return None
    path = _meta_dir() / f"{doc_id}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta
