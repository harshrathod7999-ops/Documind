"""Model warm-up so the *first user question* isn't penalised by cold model loads.

The embedding model (bge-large) and the cross-encoder reranker are lazy-loaded
on first use — that's a one-time ~20–60s cost. Without warming, it lands on the
user's first query and the chat sits on "Thinking…" long enough to feel broken.

We kick this off in a background thread at server startup (see api/main.py
lifespan) and expose readiness via ``is_ready()`` so the UI can show a one-time
"warming up" banner instead of a silent hang.
"""
from __future__ import annotations

import threading

from .retrieval import embeddings, rerank

_ready = threading.Event()
_started = threading.Event()


def _warm() -> None:
    try:
        # Loading + a tiny inference fully initialises each model.
        embeddings.embed_query("warmup")
        rerank._model().predict([["warmup query", "warmup passage"]])
    except Exception:  # pragma: no cover - never crash startup on warmup
        pass
    finally:
        _ready.set()


def start_warmup() -> None:
    """Begin loading models in the background (idempotent)."""
    if _started.is_set():
        return
    _started.set()
    threading.Thread(target=_warm, name="model-warmup", daemon=True).start()


def is_ready() -> bool:
    return _ready.is_set()
