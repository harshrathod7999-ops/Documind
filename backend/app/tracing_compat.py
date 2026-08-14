"""Bridge to the vendored ``shared.tracing`` Langfuse helpers.

DocuMind ships its own copy of ``shared/`` inside the backend so it runs
standalone. If Langfuse isn't installed/configured, the helpers degrade to
no-op decorators so the app still works.
"""
from __future__ import annotations

import sys
from pathlib import Path

# backend/ holds the vendored `shared/` package next to `app/`.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

try:
    from shared.tracing import flush, get_client, observe, tracing_enabled  # noqa: F401
except Exception:  # pragma: no cover - shared not available
    def observe(*, name=None):
        def deco(fn):
            return fn
        return deco

    def tracing_enabled() -> bool:
        return False

    def get_client():
        return None

    def flush() -> None:
        pass
