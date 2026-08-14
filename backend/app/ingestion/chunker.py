"""Structure-aware chunking (~400 tokens, ~12% overlap).

Why structure-aware rather than blind fixed-size slicing:
  * Chunks never cross page boundaries → page-level citations stay exact.
  * We detect headings and tag each chunk with its nearest section, which
    both improves retrieval (heading text is semantically loaded) and lets
    the UI show "Section 3.2 · page 14".
  * We pack by sentence so we don't cut mid-sentence, then carry ~12% token
    overlap into the next chunk so context isn't lost at the seams.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import tiktoken

from ..config import get_settings
from ..models import Chunk
from .loader import PageText

_ENC = tiktoken.get_encoding("cl100k_base")

# A line that looks like a heading: short, not ending in sentence punctuation,
# optionally numbered ("3.2 Methods") or ALL-CAPS / Title Case.
_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][^.!?]{2,80}$"
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _ntokens(text: str) -> int:
    return len(_ENC.encode(text))


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    words = line.split()
    if len(words) > 14:
        return False
    if line.endswith((".", ",", ";", ":")):
        return False
    return bool(_HEADING_RE.match(line))


@dataclass
class _Block:
    text: str
    section: str | None


def _split_blocks(page_text: str) -> list[_Block]:
    """Split a page into (paragraph, current-section) blocks."""
    blocks: list[_Block] = []
    section: str | None = None
    for para in re.split(r"\n{2,}", page_text):
        para = para.strip()
        if not para:
            continue
        # A standalone short line acts as a heading for following blocks.
        if "\n" not in para and _looks_like_heading(para):
            section = para
            continue
        blocks.append(_Block(text=para, section=section))
    if not blocks:  # whole page was one blob with no blank lines
        blocks = [_Block(text=page_text.strip(), section=section)]
    return blocks


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Take whole sentences from the end totalling ~overlap_tokens."""
    tail: list[str] = []
    total = 0
    for sent in reversed(sentences):
        t = _ntokens(sent)
        if total + t > overlap_tokens and tail:
            break
        tail.insert(0, sent)
        total += t
    return tail


def chunk_document(
    doc_id: str, doc_name: str, pages: list[PageText], source_type: str = "pdf"
) -> list[Chunk]:
    """Turn loaded pages into overlapping, page-anchored chunks."""
    settings = get_settings()
    max_tokens = settings.chunk_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    chunks: list[Chunk] = []
    idx = 0

    for page in pages:
        for block in _split_blocks(page.text):
            sentences = _sentences(block.text) or [block.text]
            cur: list[str] = []
            cur_tokens = 0

            def flush(sents: list[str]) -> None:
                nonlocal idx
                text = " ".join(sents).strip()
                if not text:
                    return
                cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
                chunks.append(
                    Chunk(
                        id=cid,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page=page.page,
                        chunk_index=idx,
                        text=text,
                        section=block.section,
                        token_count=_ntokens(text),
                        source_type=source_type,
                    )
                )
                idx += 1

            for sent in sentences:
                st = _ntokens(sent)
                # A single monster sentence: hard-split on tokens.
                if st > max_tokens:
                    if cur:
                        flush(cur)
                        cur, cur_tokens = [], 0
                    toks = _ENC.encode(sent)
                    for i in range(0, len(toks), max_tokens):
                        flush([_ENC.decode(toks[i : i + max_tokens])])
                    continue

                if cur_tokens + st > max_tokens and cur:
                    flush(cur)
                    carry = _overlap_tail(cur, overlap_tokens)
                    cur = list(carry)
                    cur_tokens = sum(_ntokens(s) for s in cur)

                cur.append(sent)
                cur_tokens += st

            if cur:
                flush(cur)

    return chunks


def new_doc_id(doc_name: str, raw: bytes) -> str:
    """Deterministic doc id from name + content hash (dedupe re-uploads)."""
    import hashlib

    h = hashlib.sha1(raw).hexdigest()[:12]
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_name}:{h}"))
