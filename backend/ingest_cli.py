"""CLI: ingest one or more PDFs into DocuMind's Qdrant collection.

    python ingest_cli.py path/to/a.pdf path/to/b.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.service import ingest_pdf


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python ingest_cli.py <file1.pdf> [file2.pdf ...]")
        return 1
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"skip (not found): {p}")
            continue
        info = ingest_pdf(str(p), p.name, p.read_bytes())
        print(f"[ok] {info.doc_name}: {info.pages} pages -> {info.chunks} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
