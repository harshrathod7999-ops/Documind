# DocuMind — The Complete Build & Mastery Guide

> A teaching guide to the project you just built: how every piece works and *why*, how to evaluate and deploy it, a full debugging playbook, and research-backed ways to push it further and **make it uniquely yours**.
>
> Model IDs and techniques below were re-verified against the live landscape in **June–July 2026** (sources at the bottom). The GenAI stack moves weekly — re-check the Groq console and Google AI Studio before relying on any model ID.

---

## 1. What we built & why it matters

**DocuMind** is a retrieval-augmented question-answering system over your own documents (PDF, DOCX, TXT, MD). You upload documents; you ask questions — including natural **follow-ups** — in plain language; it answers **grounded in the sources with clickable citations that open the original file at the cited page with the passage highlighted**, and it honestly says *"I couldn't find this in the documents"* when the answer isn't there. Every answer ships a **"How this answer was found" inspector** so you (and anyone reviewing your work) can see exactly which retrieval stage found what.

The thing that makes it portfolio-grade rather than a tutorial is the **retrieval pipeline**: instead of naive top-k vector search, it runs **dense + BM25 hybrid retrieval → Reciprocal Rank Fusion → a cross-encoder reranker → top-5**, and it proves the value of that design with a **measured before/after eval** — and now a **second eval** proving the multi-turn query-condensing step too.

**Why it matters for hiring:** retrieval engineering is the #1 in-demand GenAI skill in 2026, and the market is flooded with shallow "chat with PDF" demos. What gets interviews is *execution quality* — hybrid retrieval, a real eval harness with numbers, honest failure handling, citations you can actually click through to the source, conversational follow-ups that don't silently fail, and a live deploy. DocuMind is built to hit exactly those signals.

**Resume bullet you can now write:**
> Built a production-style RAG system (FastAPI + React + Qdrant) with hybrid retrieval (dense + BM25 + RRF) and cross-encoder reranking, lifting retrieval MRR from 0.76 → 0.89 (+0.175 on exact-match queries) on a 30-question eval set; added multi-turn follow-up handling via query condensing (MRR 0.70 → 0.90 on a second eval), a citation-to-source-PDF viewer, and a retrieval-transparency inspector exposing every pipeline stage — page-level citations, honest-refusal handling, Langfuse tracing, and Groq→Gemini failover, all on free-tier models.

---

## 2. Prerequisites

**Accounts / keys (all free, no credit card):**
- **Groq** API key — https://console.groq.com/keys (primary LLM)
- **Google AI Studio** key — https://aistudio.google.com/app/apikey (fallback LLM + RAGAS judge). *Free tier = Flash models only; Pro is paid since April 2026.*
- **Langfuse** (optional) — https://cloud.langfuse.com for tracing

**Tools:** Docker Desktop (for Qdrant), Python 3.11, Node 20+, and `uv` (fast Python package manager). All present on your machine.

**Assumed knowledge:** basic Python, a little React/TypeScript, and a conceptual grasp of embeddings (text → vector) and cosine similarity. Everything else is explained below.

---

## 3. Architecture

```
┌────────────────┐              ┌────────────────── FastAPI backend ───────────────────┐
│ React + Vite    │             │                                                        │
│  frontend       │  HTTP / SSE │  INGEST  PDF/DOCX/TXT/MD → page-aware load →           │
│                 │ ───────────▶│          structure-aware chunk (~256 tok, 15% overlap) │
│ • docs panel +  │             │          → bge-large embed → Qdrant + BM25 index       │
│   summaries     │ ◀───────────│          → cheap-LLM doc summary + suggested Qs        │
│ • chat sessions │             │                                                        │
│   (localStorage)│             │  ASK     question + chat history                       │
│ • markdown +    │             │    ├─ condense follow-up → standalone query (1 cheap   │
│   [n] cite chips│             │    │  LLM call; skipped on first turns)                │
│ • citation → PDF│             │    ├─ dense search (bge-large, Qdrant)  ─┐             │
│   viewer (page +│             │    ├─ BM25 lexical search               ─┤→ RRF fuse   │
│   highlight)    │             │    │                                     └→ rerank      │
│ • retrieval      │            │    │                        (bge-reranker-v2-m3, top-5) │
│   inspector      │            │    └─ LLM answer (history-aware) w/ [n] citations +     │
│ • warmup banner  │            │       honest refusal + per-stage retrieval trace        │
└────────────────┘              │                                                          │
                                │  LLM router (shared/llm.py): Groq gpt-oss-120b →         │
                                │                              Gemini Flash on 429/error   │
                                │  Tracing: Langfuse on every retrieval + LLM call         │
                                └──────────────────────────────────────────────────────────┘
                                          ▲                              ▲
                                   Qdrant (Docker)              Groq / Gemini (free tier)
```

**Control flow for one question** (`POST /ask` or `/ask/stream`):
1. If there's chat history, **condense the follow-up** into a standalone search query with one cheap LLM call (`generation/rewrite.py`); first turns skip this entirely.
2. Embed that query with bge-large (with the bge query-instruction prefix).
3. Dense search in Qdrant (top 20) **and** BM25 lexical search (top 20), in parallel — optionally recording a full per-stage **trace** for the UI inspector.
4. **RRF** fuses the two ranked lists into one (no score-scale normalization needed).
5. The cross-encoder reranker reads (query, passage) pairs *together* and reorders the top candidates down to **5**.
6. Those 5 chunks become a numbered `SOURCES` block; the LLM answers using only them (plus recent chat turns for tone/context), citing `[1]`, `[2]`…
7. Post-processing keeps only the citations the model actually used; the API returns answer + citations + timings + trace. The frontend renders `[n]` as clickable chips — click one and the original file opens at that page with the passage highlighted.

---

## 4. Step-by-step build (mirrors the real files)

The backend lives in [`backend/app/`](backend/app). Here's the journey of a document and a question, file by file.

### 4.1 Configuration — `app/config.py`
All tunables are env-driven via `pydantic-settings`, with sensible free-tier defaults. Key knobs:
```python
chunk_tokens: int = 256          # CHUNK_TOKENS
chunk_overlap_ratio: float = 0.15
dense_top_k: int = 20            # candidates from dense search
bm25_top_k: int = 20            # candidates from lexical search
rrf_k: int = 60                 # RRF damping constant
rerank_candidates: int = 20      # how many fused candidates to rerank
final_top_k: int = 5            # what the LLM actually sees
```
`.env` is loaded from the **project folder** (not a shared repo root) so the project runs standalone as a zip.

### 4.2 Ingestion — load → chunk → embed → store
**`ingestion/loader.py`** extracts text **per page** (page numbers are our unit of citation), de-hyphenates line breaks, and normalizes whitespace.

**`ingestion/chunker.py`** is *structure-aware*: chunks never cross a page boundary (so citations stay exact), it packs **by whole sentences** up to ~256 tokens (so we never cut mid-thought), and it carries **~15% token overlap** into the next chunk so context isn't lost at the seams. Token counts use `tiktoken`. The interview-relevant idea:
```python
# pack sentences until we'd exceed the budget, then flush and carry overlap
if cur_tokens + st > max_tokens and cur:
    flush(cur)
    carry = _overlap_tail(cur, overlap_tokens)   # keep whole trailing sentences
    cur, cur_tokens = list(carry), sum(_ntokens(s) for s in carry)
```

**`retrieval/embeddings.py`** uses the local `BAAI/bge-large-en-v1.5` model (free, unlimited, CPU). Note the **asymmetry**: bge wants an instruction prefix on *queries* but not on *passages* — honoring that measurably helps recall.
```python
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# passages embedded raw; queries embedded with the prefix
```

**`retrieval/store.py`** wraps Qdrant: creates the collection (cosine, 1024-dim), upserts points with the full chunk payload, runs dense `query_points`, and scrolls all chunks (used to rebuild BM25). Payload **is** our source of truth.

### 4.3 The retrieval pipeline (the differentiator) — `retrieval/pipeline.py`
Three stages, three modes (the modes exist so the eval can isolate each stage's contribution):

**BM25 (`retrieval/bm25.py`)** — a lexical index rebuilt in-memory from Qdrant payloads. It catches what dense vectors are worst at: exact tokens like `E-4021`, `SKU-9981`, `192.168.10.1`.

**RRF fusion** — combines the dense and BM25 rankings by rank position, not score, so the incomparable cosine and BM25 scales don't matter:
```python
score[chunk] += 1.0 / (rrf_k + rank)   # summed across both lists; rrf_k = 60
```

**Reranking (`retrieval/rerank.py`)** — `BAAI/bge-reranker-v2-m3` is a cross-encoder: it reads the query and passage **together**, so it's far more precise than the bi-encoders that produced the candidates. We over-retrieve (~20 fused) and rerank down to 5. This is the single biggest precision gain in the pipeline.

### 4.4 Generation with citations — `generation/answer.py`
The system prompt enforces the contract: answer **only** from the numbered sources, cite with inline `[n]` markers, surface conflicts, and refuse honestly when the answer isn't present. After generation we parse the markers the model actually used and build the citation objects — so the UI never shows a dangling source.

### 4.5 The shared LLM router — `shared/llm.py` (vendored)
Every LLM call goes through one `chat()` / `chat_stream()`. It tries **Groq `gpt-oss-120b`**, retries transient errors with **exponential backoff + jitter**, and **falls back to Gemini Flash** on rate limits — so a Groq 429 doesn't kill the demo. Langfuse tracing is wired into LiteLLM's callbacks automatically. This file is *vendored into the project* (not imported from a repo root) so the zip is self-contained.

### 4.6 The API — `app/api/main.py`
FastAPI with `/ingest`, `/documents`, `/ask`, `/ask/stream` (SSE: sources frame → token frames → citations frame), and `/health`. A **lifespan hook** kicks off model warm-up in a background thread at startup (`app/warmup.py`) so the embedder + reranker cold-load (~20–60s) happens *before* the user's first question — and `/health` exposes `models_ready` so the UI can show a one-time banner.

### 4.7 The frontend — `frontend/src/`
React 19 + Vite + Tailwind v4 + full **shadcn/ui** (dialog, sheet, dropdown-menu, tabs, sonner toasts, tooltip, skeleton — hand-wired into the project's existing HSL design tokens rather than run through `shadcn init`, so the original theme survived; see §5). `lib/api.ts` is a typed client incl. a hand-rolled SSE-over-POST parser (EventSource can't POST). `ModelStatusBanner.tsx` polls `/health` and shows the warm-up notice. State: `@tanstack/react-query` for server data, `zustand` (+ `persist`) for chat sessions and preferences. A `ThemeToggle` drives light/dark/system by toggling `<html class="dark">`, and `useMediaQuery`/`useKeyboardShortcuts` hooks handle responsive breakpoints and the `/`-to-focus, `Esc`-to-close shortcuts.

### 4.8 Citation → PDF viewer — the headline UX upgrade
Citations pointed at a page number, but nothing let you *see* the page — a real gap for a "document intelligence" tool. Two pieces close it:

**Backend** — `GET /documents/{doc_id}/file` in `app/api/main.py` resolves the doc_id to its stored file and returns a `FileResponse` (Starlette handles HTTP Range requests for free, which `react-pdf` relies on for fast page seeking). This required fixing a latent bug first: uploads were stored under their **original filename**, so two different files named e.g. `report.pdf` would silently overwrite each other's bytes on disk while both stayed listed (with different doc_ids) in Qdrant — a citation could then open the *wrong* file. Fix: compute the doc_id (already a deterministic `uuid5` of name+content-hash) **before** writing, and store as `{doc_id}{ext}`.

**Frontend** — `PdfViewerPanel.tsx` uses `react-pdf` (a React wrapper over pdf.js). The classic Vite trap here is a pdf.js **worker version mismatch**: you must resolve the worker file *from the same `pdfjs-dist` copy react-pdf bundles*, not a separately installed one:
```ts
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url
).toString();
```
The viewer opens at the cited page and does **best-effort snippet highlighting** via pdf.js's `customTextRenderer`: it tries progressively shorter word-prefixes of the citation snippet against each text-layer fragment (pdf.js splits page text into short line fragments, so the whole snippet rarely appears verbatim in one fragment) and wraps a match in `<mark>`. If nothing matches, the page still opens — highlighting is a nice-to-have, not a blocking feature. The component is **lazy-loaded** (`React.lazy`) because pdf.js is ~1 MB minified; it only enters the bundle the first time a user clicks a citation.

### 4.9 Multi-turn conversational RAG — `generation/rewrite.py`
Follow-up questions ("How do I fix it?") retrieve terribly on their own — the key terms live in the *previous* turn, not the follow-up itself. `condense_question(question, history)` spends **one cheap LLM call** rewriting the follow-up into a standalone search query before retrieval runs:
```python
if not history:
    return question, False          # first turns: zero extra cost, zero extra latency
...
rewritten = chat(messages, model=settings.rewrite_model,   # gpt-oss-20b — small & fast
                  temperature=0.0, max_tokens=600)
```
Two non-obvious things earned their place here the hard way (see §5): the `max_tokens=600` (not a tight 120) and the fact the fallback is unconditional — any exception, empty result, or degenerate output (multi-line, absurdly long) falls straight back to the raw question. **This feature can only degrade `/ask`'s quality on a bad turn, never break it.** The condensed query flows into both retrieval *and* an `event: meta` SSE frame so the UI can show "rewritten from follow-up."

Generation itself becomes history-aware in `generation/answer.py`: `_build_messages()` now interleaves up to 6 recent chat turns between the system prompt and the fresh `SOURCES` block, so the model has conversational context (what "it" refers to) while still being *grounded* only in the current retrieval — history informs tone/continuity, not facts.

### 4.10 Retrieval-transparency inspector
The hybrid pipeline is DocuMind's whole thesis, but a user (or a recruiter demoing it) previously had no way to *see* it working. `retrieve_traced()` in `retrieval/pipeline.py` runs the identical hybrid_rerank pipeline while recording each stage's top candidates and timings into a small `RetrievalTrace` — kept intentionally light (top-10 per stage, `chunk_id`/`doc_name`/`page`/`rank`/`score` only, no chunk text) so it adds a few KB to the SSE stream, not a resend of the whole context. The frontend's `RetrievalInspector.tsx` renders it as a collapsible "How this answer was found" panel with tabs for each stage, and — the best interview talking point in the whole feature — computes a **"rescued by BM25" badge** client-side: any chunk_id that appears in the reranked finalists *and* in the BM25 candidates *but not* in the dense candidates is BM25's save, proven per-answer instead of asserted in a README.

### 4.11 Broader ingestion + per-document intelligence
`ingestion/loader.py` gained a `load_document(path, filename) -> (pages, source_type)` dispatcher alongside the original `load_pdf`: `load_docx` (python-docx, heading-styled paragraphs kept as standalone lines so the *existing* heading-regex section detector fires unchanged) and `load_text` (TXT/MD, with markdown `#` headers stripped to plain lines for the same reason). Both synthesize **pseudo-pages** (~4,000 chars) since DOCX/TXT/MD have no physical pages — the chunker, citation model, and UI all treat this as `source_type` (`"pdf" | "docx" | "text"`), which is what gates the PDF viewer (non-PDF citations show their snippet in a card instead of trying to open a page that doesn't exist).

At the end of ingest, one more cheap LLM call (`ingestion/docmeta.py`) generates a short summary + 3 suggested questions, stored as a **sidecar JSON** file (`data/meta/{doc_id}.json`) rather than duplicated onto every chunk's Qdrant payload — a deliberate storage choice (see §5). `/documents` merges the sidecar in at read time. The frontend surfaces the summary on each doc card and uses the suggested questions as the empty-state's clickable prompts instead of static placeholder text — which is exactly the behavior you're seeing if you upload, say, an annual report: the questions shown will be about *that* document, generated at ingest time, not hardcoded in the app.

### 4.12 Chat sessions
`store.ts` restructures chat state from a flat `messages[]` into `sessions: Record<id, ChatSession>` + `activeSessionId`, wrapped in zustand's `persist` middleware writing to `localStorage`. The persisted `partialize` strips the (large, ephemeral) `trace` field and any `streaming: true` flags before saving, and caps sessions/messages to bound storage growth. `SessionsMenu.tsx` in the header exposes new/switch/rename/delete.

---

## 5. The hard parts & how we solved them

| Hard part | What went wrong | Fix |
|-----------|-----------------|-----|
| **Tiny eval corpus made retrieval trivial** | 400-token chunks collapsed the toy corpus to ~8 chunks; `top-k=5` returned almost everything, so every mode scored ~100% — no measurable lift. | Moved to **256-token chunks** + a larger, distractor-rich corpus (a near-identical *X100* product) so `top-k` actually discriminates. The lift became visible, concentrated in exact-match queries. |
| **First question hung on "Thinking…"** | bge + reranker lazy-load on first use (~20–60s); it landed on the user's first query and felt broken. | **Warm models at startup** in a background thread + a **`models_ready` flag** on `/health` driving a one-time UI banner. |
| **Standalone distribution** | Code reached "up" to a repo-root `shared/`; zipping one project would break it. | **Vendored `shared/`** into the project; verified it imports with the repo root hidden. |
| **Rate limits (429)** | Groq free tier is ~30 RPM / 1K RPD. | Central **backoff + Groq→Gemini fallback** in the router; shrink eval batches rather than switching to paid. |
| **Windows console crash** | Printing `✓`/`→` crashed on cp1252. | ASCII-safe logs + `PYTHONIOENCODING=utf-8` for scripts. |
| **Dependency conflict** | RAGAS pulls an older `langchain-core` than the serving stack. | **Split** `requirements.txt` (serving) from `requirements-eval.txt` (offline eval, separate venv). |
| **`gpt-oss` returned an empty string for the query-rewrite call** | `gpt-oss` models are **reasoning models** — they spend tokens on internal reasoning *before* emitting visible content. A tight `max_tokens=120` (plenty for a short query, we assumed) got entirely consumed by reasoning, so the visible completion was `""`. | Raised the cheap-call budgets to `600`/`800` tokens. Caught by testing the rewrite call directly against the live model rather than trusting the "it looks fine, it's just a query rewrite" assumption. |
| **Citations silently stopped working with `gpt-oss`** | Some completions came back with fullwidth CJK brackets (`【1】`) instead of ASCII `[1]`, despite the prompt explicitly specifying `[n]` — so the citation-marker regex found nothing and every answer looked ungrounded. | Normalize per-token with `str.translate({"【":"[","】":"]"})` in both the sync and streaming generation paths — safe because each bracket is a single character, so it holds even split across streamed token boundaries. |
| **Two files with the same name silently overwrote each other** | Uploads were stored on disk by original filename; a second `report.pdf` clobbered the first's bytes while Qdrant kept both (different doc_ids) — a citation could then open the wrong document. | Store files as `{doc_id}.{ext}` (doc_id is a content-hash-based `uuid5`, computed *before* the write) — see §4.8. |
| **pdf.js worker mismatch under Vite** | Installing `pdfjs-dist` separately from `react-pdf` (which bundles its own copy) threw "API version does not match Worker version" at runtime. | Resolve the worker from react-pdf's own bundled `pdfjs-dist` via `new URL(..., import.meta.url)`; never install a second copy. |
| **Markdown rendering would swallow `[n]` citation markers** | `react-markdown` parses `[1]` as a potential link-reference syntax edge case, and any post-processing of *rendered HTML* breaks the moment a marker sits inside bold/italic/list-item text. | A **custom remark plugin** operating on the mdast **text nodes** (before HTML rendering) splits on `[n]` and emits a custom `cite` node — this naturally skips code blocks (their content isn't a `text` node) and survives nesting inside any other markdown construct. |
| **Which model rewrites follow-ups, and what if it's wrong?** | A wrong or slow rewrite could make multi-turn strictly worse than doing nothing. | The rewrite call uses the cheap/fast model (`gpt-oss-20b`), has a hard `try/except` → raw-question fallback, and a "sanity" check (reject empty, multi-line, or absurdly long completions) before trusting it — the eval in §6 measures the *actual* lift, not an assumed one. |

---

## 6. Evaluation & results

The eval set is 30 questions ([`backend/eval/dataset.json`](backend/eval/dataset.json)) mixing **exact-match** (codes/IDs), **conceptual/paraphrased**, **conflicting-source**, and **unanswerable** questions over a 3-document corpus. Retrieval metrics need no API key (local embeddings); generation metrics use a free Gemini Flash judge via RAGAS.

**Real run (top-k=5, 256-token chunks):**

| Mode | Hit@5 | MRR |
|------|------:|----:|
| Dense only (baseline) | 96.2% | 0.756 |
| Hybrid (dense+BM25, RRF) | 96.2% | 0.772 |
| **Hybrid + rerank (DocuMind)** | **100.0%** | **0.894** |

**Where the lift concentrates (MRR by type):**

| Question type | Dense | Hybrid+rerank | Δ |
|---|---:|---:|---:|
| Exact-match (codes/IDs) | 0.725 | 0.900 | **+0.175** |
| Conceptual / paraphrased | 0.743 | 0.875 | +0.132 |
| Conflicting sources | 1.000 | 1.000 | +0.000 |

The story: dense search ranks the *X100* distractor passage near the correct *X200* one for a bare code like `E-4021`; BM25 + the cross-encoder pull the right chunk to rank 1. That is the entire argument for hybrid retrieval, quantified.

Reproduce: `python eval/make_sample_docs.py && python ingest_cli.py eval/sample_docs/*.pdf && python eval/run_eval.py --no-ragas`.

**Multi-turn eval** ([`eval/multiturn_dataset.json`](backend/eval/multiturn_dataset.json), `eval/run_multiturn_eval.py`) — 10 short conversations whose follow-ups are deliberately underspecified ("And after that window?"), comparing retrieval on the **raw follow-up** vs the **condensed standalone query** from §4.9:

| Query used | Hit@5 | MRR |
|------------|------:|----:|
| Raw follow-up | 70.0% | 0.700 |
| **Condensed (DocuMind)** | **90.0%** | **0.900** |

**Lift from condensing: Hit@5 +20 pts, MRR +0.200.** Same structure as the retrieval eval on purpose — this is the "did the feature I added actually help, measured, not asserted" habit applied a second time. Reproduce: `python eval/run_multiturn_eval.py` (needs a Groq/Gemini key for the condense call).

---

## 7. Deployment (free tier)

- **Vector store:** Qdrant Cloud free tier (1 GB). Set `QDRANT_URL` + `QDRANT_API_KEY`.
- **Backend:** Render or Hugging Face Spaces using [`backend/Dockerfile`](backend/Dockerfile) (self-contained — `docker build -t documind-api .` from `backend/`). The image pre-downloads the models at build time so deploys only pay the in-memory load, not a download.
- **Frontend:** Vercel. Set `VITE_API_BASE` to the deployed backend URL; the backend's `CORS_ORIGINS` must include the Vercel domain.
- **Tracing:** point `LANGFUSE_*` at Langfuse cloud; every call shows up with latency/cost.

> Free-tier cold starts: Render free dynos sleep; the first request after sleep pays the startup + warm-up. The `models_ready` banner already communicates this to users.

> The PDF viewer (`react-pdf`/pdf.js, ~1 MB) is code-split via `React.lazy` — it's absent from the main bundle until a user actually opens a citation, so it costs nothing on first load or on devices that never click one.

---

## 8. Debugging playbook (find it → fix it)

A symptom-driven checklist. Each item: **how to confirm**, then **how to fix**.

### Backend won't answer / generic errors
- **`/health` shows `qdrant_ok: false`** → Qdrant is down. Confirm: `docker ps` (look for `documind-qdrant`) and `curl http://localhost:6333/healthz`. Fix: `docker compose up -d` from `1-documind/`; make sure **Docker Desktop is actually running** (on Windows the engine pipe disappears when it's stopped).
- **`WinError 10061 / connection refused`** in tracebacks → same root cause (Qdrant/daemon down). Fix as above.
- **`models_ready` stuck `false` for minutes** → first run is **downloading** model weights from Hugging Face (bge-large ~1.3 GB, reranker ~600 MB). Confirm: watch the uvicorn log for HF download lines, or check `~/.cache/huggingface`. Fix: just wait once; subsequent starts only *load* (~20–30s). Set `HF_HOME` to a persistent path on deploy.

### Answers are wrong / empty / "not in the documents" for things that ARE there
- **Confirm what's indexed:** `curl http://127.0.0.1:8000/health` → `indexed_chunks`. If 0, nothing was ingested.
- **Inspect retrieval directly** (no LLM needed):
  ```python
  from app.retrieval.pipeline import retrieve
  for h in retrieve("your question", mode="hybrid_rerank", top_k=5):
      print(round(h.score,3), h.chunk.page, h.chunk.text[:80])
  ```
  If the right chunk isn't in the list → a *retrieval* problem (chunking/embedding). If it *is* there but the answer is wrong → a *generation/prompt* problem.
- **PDF ingested but few/no chunks** → the PDF is **scanned images** (no text layer). `pypdf` returns empty; you'll get the explicit "no extractable text" error. Fix: add OCR (see §9 stretch goals) or use a text PDF.
- **Exact-match query fails** → BM25 may be stale. The index rebuilds after `bm25.invalidate()` on ingest; if you wrote to Qdrant out-of-band, restart the server.

### Rate limits / LLM failures
- **`RateLimitError` / 429** → Groq free tier exhausted (~30 RPM, 1K RPD). The router auto-backs-off then fails over to Gemini Flash. Confirm failover in Langfuse traces. Fix for eval: shrink the batch or add `time.sleep`; never switch to a paid model.
- **`/ask` errors but retrieval works** → missing/invalid `GROQ_API_KEY` *and* `GEMINI_API_KEY`. Both providers failed. Check keys are set as env vars or in `.env` in `1-documind/` and are valid.
- **Follow-up condensing silently does nothing (`rewritten: false` on an obvious follow-up)** → the rewrite call to `gpt-oss-20b` returned empty/degenerate output and the code correctly fell back to the raw question (by design — see §5). Confirm: call `condense_question()` directly with a generous `max_tokens` and print the raw output; if it's `""`, the token budget is too tight for a reasoning model — see the "empty string" row in §5.
- **Citations vanish (`grounded: false`) even though the answer clearly cites sources** → check the raw model output for fullwidth `【…】` brackets instead of `[…]` (a `gpt-oss` quirk, see §5). The bracket-normalization fix should already be applied in `generation/answer.py`; if you changed models, verify the new model's actual bracket style.
- **Per-document summary / suggested questions are always `null`** → the docmeta LLM call failed or returned unparseable JSON (it degrades silently by design, never fails ingest). Check the model has enough `max_tokens` headroom and that `chat()` isn't raising upstream; re-upload the doc after fixing.

### Citation viewer
- **Clicking a `[n]` chip does nothing / opens a blank panel** → check `source_type` on that citation: only `"pdf"` citations open the viewer (DOCX/TXT/MD show their snippet in the card instead — this is intentional, not a bug).
- **"Couldn't load the original file" in the viewer** → the file was ingested *before* the doc_id-based storage fix (§4.8/§5) and its bytes are gone; check `backend/data/uploads/{doc_id}.*` exists. Fix: re-upload the document.
- **pdf.js "API version does not match Worker version"** → a second `pdfjs-dist` got installed independently of `react-pdf`'s bundled copy. Remove any standalone `pdfjs-dist` dependency; the worker must be resolved from `react-pdf`'s own copy (see §4.8/§5).
- **Highlight doesn't appear on the cited page** → expected sometimes — highlighting is best-effort prefix matching against pdf.js's short text-layer fragments; the page still opens correctly. Not a bug to chase.

### Frontend issues
- **CORS error in the browser console** → the backend `CORS_ORIGINS` doesn't include your frontend origin. Fix: add `http://localhost:5173` (dev) or the Vercel URL (prod) to `CORS_ORIGINS` and restart.
- **Frontend can't reach API / 404s** → in dev, Vite proxies `/api` → `:8000` (see `vite.config.ts`); make sure the backend is on 8000. In prod, set `VITE_API_BASE` to the backend URL.
- **Tokens don't stream (answer appears all at once)** → a proxy is buffering SSE. The backend already sends `X-Accel-Buffering: no`; if you put nginx/Cloudflare in front, disable response buffering for `/ask/stream`.
- **`localhost` returns a weird JSON error but `127.0.0.1` works** → `localhost` resolved to IPv6 `::1` and hit a different listener. Use `127.0.0.1` or bind uvicorn to both.

### Eval / tooling
- **`ModuleNotFoundError: fpdf` / ragas** → those are **eval** deps. `uv pip install -r requirements-eval.txt` (ideally in a separate `.venv-eval`).
- **`UnicodeEncodeError: '✓'`** on Windows → set `PYTHONIOENCODING=utf-8` before running scripts.
- **RAGAS skipped** → no `GEMINI_API_KEY`. Set it, or run `--no-ragas` for retrieval-only metrics.

### General technique
Bisect the pipeline: **ingestion → retrieval → generation**. Print at each boundary (`indexed_chunks`, the retrieved chunk texts, the final prompt). 90% of "the AI is dumb" bugs are actually retrieval bugs you can see by looking at the chunks.

---

## 9. Make it uniquely *yours* (and push it past tutorial-grade)

Recruiters see a thousand "chat with PDF" repos. Two ways to stand out: **pick a real domain**, and **add a technique most demos skip**. Pick 1–2, not all.

> Since this list was written, DocuMind already picked up several of the highest-ROI additions on its own: multi-turn query condensing (a form of query transformation, §4.9), a citation-to-source viewer, a retrieval-transparency inspector, broader file-format support, and per-document summaries. The remaining items below are still genuinely open — none of them are done yet.

### A. Re-skin it for a domain you care about (highest ROI for "unique")
Swap the toy corpus for a real vertical and build a *real* 30-Q eval set for it. The code barely changes; the story changes completely:
- **Legal** — contract clause Q&A ("what's the termination notice period?") with clause-level citations.
- **Healthcare** — clinical-guideline lookup with strict honest-refusal (safety-critical).
- **Finance** — 10-K / annual-report analysis; handle **tables** (most RAG ignores them).
- **Internal wiki / onboarding bot** — your company's docs; add metadata filters by team.
- **Developer docs** — answer over a library's docs with code-aware chunking.
- **Support deflection** — search past tickets and draft replies (an explicitly employer-valued use case).

Frame the README around an **outcome and metric** for that domain ("reduced average answer-lookup time / cut retrieval failures by X%"), not a tool list.

### B. Add a standout retrieval technique (each is a strong interview talking point)
1. **Contextual Retrieval (Anthropic)** — *highest-impact upgrade.* Before embedding each chunk, use a cheap free model (Gemini Flash) to generate a 1–2 sentence summary situating the chunk in the whole document, and **prepend it** to the chunk. Published results: **~49% fewer retrieval failures** (up to ~67% combined with reranking). It's perfectly on-brand (uses your free router) and directly extends your existing pipeline.
2. **Move BM25 into Qdrant (native sparse vectors / FastEmbed)** — replace the in-memory `rank-bm25` with Qdrant's native sparse-vector + BM25 support and the **Query API's** server-side fusion. This fixes your documented "in-memory BM25" limitation, makes it persistent and scalable, and is a clean "I understand the vector DB deeply" signal.
3. **Semantic chunking** — split where adjacent-sentence embedding similarity drops below a threshold instead of fixed sizes; reported accuracy lifts over fixed-size baselines. Compare it against your current chunker *in the eval* — the comparison is the portfolio gold.
4. **Late-interaction reranking (ColBERT / answerai-colbert)** — near cross-encoder accuracy at near bi-encoder speed; or add **SPLADE** for a third (learned-sparse) retrieval signal toward a "Blended RAG" setup.
5. **Query transformation for vague single-turn questions** — add HyDE (hypothetical-answer embedding) or multi-query fan-out/merge for questions that are standalone but underspecified. (Different from the follow-up condensing already built in §4.9, which solves *conversational* under-specification, not single-turn vagueness — a good distinction to be able to draw in an interview.)
6. **Metadata filtering + multi-tenancy** — per-document/-team filters and isolated collections; demonstrates production concerns employers explicitly look for.

### C. Productionization that signals seniority
7. **CI eval gate** — run the eval in GitHub Actions and **fail the build** if faithfulness/recall regresses past a threshold (DeepEval or RAGAS). This is rare in portfolios and highly valued — and it's literally Project 5's theme, so you can preview it here.
8. **Answer + embedding caching** (Redis/SQLite) for repeated questions; show the latency win.
9. **Structured-document handling** — parse tables/figures (e.g. a layout-aware parser) so it works on real reports, not just prose.
10. **Feedback loop** — thumbs up/down on answers, logged to Langfuse, feeding a growing eval set.

> Suggested sequence for maximum signal with minimum sprawl: **(A) pick your domain → (B1) Contextual Retrieval → run the eval to show the lift → (C7) CI eval gate → deploy.** That's a coherent, senior-looking story.

---

## 10. How to talk about it in an interview

**Q: Why hybrid retrieval instead of just vector search?**
> Dense embeddings are great at meaning but weak at literal tokens — IDs, error codes, SKUs. BM25 nails those. I fuse both with Reciprocal Rank Fusion and rerank with a cross-encoder. My eval shows the lift is concentrated exactly where you'd predict: exact-match queries went from 0.72 to 0.90 MRR.

**Q: Why RRF and not weighted score blending?**
> Cosine and BM25 scores aren't on the same scale, and normalizing them is brittle. RRF only uses rank position, so it's robust and has one parameter. It's the standard fusion method for a reason.

**Q: How do you know it actually works?**
> A 30-question eval set spanning exact-match, conceptual, conflicting, and unanswerable questions, scored in three retrieval modes so I can attribute the gain to each stage — plus RAGAS faithfulness and honest-refusal accuracy. The numbers are in the README, not vibes.

**Q: How do you stop it from hallucinating?**
> The prompt restricts answers to the numbered sources and forces an explicit refusal when the answer isn't present — and that refusal is a *scored* eval case, not an afterthought. Reranking also reduces hallucination by feeding the model better context.

**Q: What were the trade-offs?**
> Chunk size (precision vs index size — I chose 256 tokens for citation precision), in-memory BM25 (simple now, would move to Qdrant-native sparse at scale), and local models (free + private, slightly slower than hosted). I can point to each decision in the README's Decisions section.

**Q: How do you handle follow-up questions in a RAG system?**
> Retrieval needs a standalone query, but a natural follow-up like "how do I fix it?" doesn't have one on its own — the key terms are in the previous turn. I spend one cheap LLM call condensing the follow-up into a standalone search query before retrieval, then feed the recent conversation to the generator for tone/context while still grounding facts only in the freshly retrieved sources. I measured it: MRR went from 0.70 retrieving on the raw follow-up to 0.90 on the condensed query across 10 test conversations.

**Q: What if that rewrite step is wrong or the model call fails?**
> It degrades, it never breaks. First turns skip the call entirely — zero added cost or latency. On any turn with history, a failed call, an empty response, or a suspicious output (multi-line, absurdly long) falls straight back to the raw question. Worst case you're back to normal single-turn retrieval quality; it can't make things worse.

**Q: Why store chat sessions in the browser instead of a database?**
> This ships as a self-contained zip — a single-user demo tool. A backend database would mean schema, migrations, and CRUD endpoints for a feature that, for this use case, localStorage handles in about 100 lines via zustand's persist middleware. The honest trade-off is chats don't sync across devices, which I call out explicitly rather than pretend doesn't exist.

**Q: How do you show users *why* an answer was retrieved, not just what it says?**
> Every answer carries an optional retrieval trace — the top candidates and scores at each pipeline stage (dense, BM25, RRF fusion, rerank), plus stage timings — streamed alongside the answer. The UI renders it as a collapsible panel, and computes a "rescued by BM25" badge client-side for any final citation that BM25 found but dense search missed. It turns an abstract architecture claim into something you can point at on a specific answer.

**Q: Tell me about a bug you found through your own eval/testing, not from a user report.**
> While wiring up the follow-up-rewrite feature, I tested the raw LLM call directly and got back an empty string. `gpt-oss` models reason internally before producing visible output, and my `max_tokens=120` (which felt generous for a short query) was entirely consumed by that reasoning — so nothing ever became visible content. I caught it by testing the isolated call rather than trusting "it's just a rewrite, it'll be fine," and the fix was simply raising the budget to 600 tokens with a comment explaining why the number looks oddly large for such a short output.

---

## 11. Reference: current free models (re-verify before relying on these)

| Role | Model | Notes (June–July 2026) |
|------|-------|-------------------|
| Primary LLM | Groq `openai/gpt-oss-120b` | Strong free reasoning; ~30 RPM / 1K RPD free. Reasoning model — see §5 on `max_tokens`. |
| Cheap/fast calls (rewrite, doc summaries) | Groq `openai/gpt-oss-20b` | Query condensing + per-doc summaries; give it ≥600 `max_tokens` (reasoning overhead). |
| Alt open LLM | Groq `qwen/qwen3.6-27b` | Recommended migration target on Groq. |
| Fallback / multimodal | Gemini `gemini-3.5-flash` | Free tier is **Flash-only**; Gemini **Pro is paid** since Apr 2026. ~1,500 RPD free. |
| Embeddings | local `BAAI/bge-large-en-v1.5` | Free, unlimited, CPU. (Gemini `gemini-embedding-001` is a free API alternative.) |
| Reranker | local `BAAI/bge-reranker-v2-m3` | Cross-encoder, CPU, no paid Cohere. |

> ⚠️ Groq deprecated `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` (June 2026). Always confirm live IDs in the [Groq console](https://console.groq.com/docs/models) and [Google AI Studio](https://aistudio.google.com).

---

## Sources / further reading
- [Groq supported models](https://console.groq.com/docs/models) · [Groq free-tier limits 2026](https://tokenmix.ai/blog/groq-free-tier-limits-2026) · [GPT-OSS-120B on Groq](https://console.groq.com/docs/model/openai/gpt-oss-120b)
- [Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) · [Gemini free tier 2026](https://tokenmix.ai/blog/gemini-api-free-tier-limits)
- [Anthropic Contextual Retrieval — DataCamp guide](https://www.datacamp.com/tutorial/contextual-retrieval-anthropic) · [Together AI implementation](https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic)
- [RAG best practices 2026](https://www.callmissed.com/en/blog/rag-best-practices-2026) · [Advanced retrieval patterns 2026](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo) · [12 advanced RAG techniques](https://atlan.com/know/advanced-rag-techniques/)
- [Qdrant hybrid search (Query API)](https://qdrant.tech/articles/hybrid-search/) · [Qdrant hybrid + reranking tutorial](https://qdrant.tech/documentation/tutorials-search-engineering/reranking-hybrid-search/)
- [react-pdf (wojtekmaj) docs](https://github.com/wojtekmaj/react-pdf) · [pdf.js worker setup](https://github.com/wojtekmaj/react-pdf#configure-pdfjs-worker)
- [GenAI portfolio projects that get hired 2026](https://idolsrm.in/ai-portfolio-projects-2026/) · [RAG engineer roadmap + projects 2026](https://jober.jankariweb.online/rag-engineer-roadmap-2026-portfolio-projects-interview-tips/)
