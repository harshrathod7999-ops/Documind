# DocuMind — RAG Document Intelligence

Ask natural-language questions across your documents (**PDF, DOCX, TXT, MD**) and get answers **grounded in the source with page-level citations** — click any citation to open the **original PDF at the cited page with the passage highlighted**. Follow-up questions work (**multi-turn conversational RAG** with query condensing), every answer ships a **"How this answer was found" inspector** showing the dense/BM25/RRF/rerank pipeline stages, and the model says an honest "not in the documents" when the answer isn't there. Built on **free models only** (local embeddings/reranker + Groq/Gemini free tier).

> **Why this project exists:** retrieval engineering is the #1 in-demand GenAI skill, and most demos ship naive top-k vector search that quietly fails on exact-match queries (IDs, codes, SKUs) and surfaces irrelevant chunks. DocuMind ships the **production pattern**: hybrid retrieval → RRF fusion → cross-encoder reranking, measured with a real eval set that quantifies the lift.

---

## Architecture

```
┌────────────────┐     ┌──────────────────────── FastAPI backend ──────────────────────────┐
│ React + Vite   │     │                                                                    │
│  chat sessions │     │  INGEST:  PDF/DOCX/TXT/MD → page-aware load → structure-aware      │
│  (persisted)   │     │           chunk (~256 tok, 15% overlap) → bge-large embed →        │
│  docs panel    │ HTTP│           Qdrant + BM25 index → doc summary + suggested questions  │
│  drag-drop     │────▶│                                                                    │
│  upload        │ SSE │  ASK:     question + chat history                                  │
│  markdown      │◀────│     └─ condense follow-up → standalone query (cheap LLM, 1 call)   │
│  answers with  │     │     ├─ dense search (bge-large, Qdrant)  ─┐                        │
│  [n] chips     │     │     ├─ lexical search (BM25)             ─┤→ RRF merge → top-20    │
│  PDF viewer    │     │                                            └→ bge-reranker-v2-m3   │
│  (cited page + │     │                                               → top-5 → LLM answer │
│  highlight)    │     │                                                  with [n] citations│
│  retrieval     │     │           + per-stage trace (dense/bm25/rrf/rerank) for the UI     │
│  inspector     │     │  LLM via shared/llm.py: Groq gpt-oss-120b → Gemini Flash fallback  │
└────────────────┘     │  Tracing: Langfuse (every retrieval + LLM call)                    │
                       └────────────────────────────────────────────────────────────────────┘
```

**Retrieval pipeline (the differentiator)** — [`app/retrieval/pipeline.py`](backend/app/retrieval/pipeline.py):

1. **Dense** — `BAAI/bge-large-en-v1.5` embeddings in **Qdrant** (cosine). Query gets the bge query-instruction prefix; passages don't (asymmetric, as the model intends).
2. **Lexical** — **BM25** over the same chunks. Catches exact tokens dense vectors miss: `E-4021`, `SKU-9981`, `192.168.10.1`.
3. **RRF merge** — Reciprocal Rank Fusion (`score = Σ 1/(k+rank)`), which combines the two rankings without needing comparable score scales.
4. **Rerank** — `BAAI/bge-reranker-v2-m3` cross-encoder reads (query, passage) *together* and reorders the top candidates down to the final 5.
5. **Generate** — grounded answer with inline `[n]` citations mapped to page numbers; honest refusal when retrieval is empty or off-topic.

---

## What's inside

| Area | Tech | File(s) |
|------|------|---------|
| Document load (PDF/DOCX/TXT/MD, page-aware) | pypdf, python-docx | [`ingestion/loader.py`](backend/app/ingestion/loader.py) |
| Structure-aware chunking | tiktoken | [`ingestion/chunker.py`](backend/app/ingestion/chunker.py) |
| Per-doc summary + suggested questions | cheap LLM call, sidecar JSON | [`ingestion/docmeta.py`](backend/app/ingestion/docmeta.py) |
| Dense embeddings | sentence-transformers (bge-large) | [`retrieval/embeddings.py`](backend/app/retrieval/embeddings.py) |
| Vector store | Qdrant | [`retrieval/store.py`](backend/app/retrieval/store.py) |
| Lexical retrieval | rank-bm25 | [`retrieval/bm25.py`](backend/app/retrieval/bm25.py) |
| Reranking | bge-reranker-v2-m3 cross-encoder | [`retrieval/rerank.py`](backend/app/retrieval/rerank.py) |
| Fusion + orchestration + stage trace | RRF | [`retrieval/pipeline.py`](backend/app/retrieval/pipeline.py) |
| Follow-up query condensing (multi-turn) | cheap LLM call w/ hard fallback | [`generation/rewrite.py`](backend/app/generation/rewrite.py) |
| Grounded generation (history-aware) | shared LiteLLM router | [`generation/answer.py`](backend/app/generation/answer.py) |
| API (incl. SSE streaming + file serving) | FastAPI | [`api/main.py`](backend/app/api/main.py) |
| Frontend | React 19 + Vite + Tailwind v4 + shadcn/ui | [`frontend/`](frontend/) |
| Citation PDF viewer (page + highlight) | react-pdf (lazy-loaded) | [`frontend/src/components/PdfViewerPanel.tsx`](frontend/src/components/PdfViewerPanel.tsx) |
| Markdown answers w/ citation chips | react-markdown + custom remark plugin | [`frontend/src/lib/remarkCitations.ts`](frontend/src/lib/remarkCitations.ts) |
| Retrieval inspector UI | per-stage trace over SSE | [`frontend/src/components/RetrievalInspector.tsx`](frontend/src/components/RetrievalInspector.tsx) |
| Chat sessions (persisted) | zustand persist → localStorage | [`frontend/src/store.ts`](frontend/src/store.ts) |
| Eval harness | retrieval metrics + RAGAS + multi-turn | [`backend/eval/`](backend/eval/) |

---

## Evaluation & results

30-question eval set ([`backend/eval/dataset.json`](backend/eval/dataset.json)) over a 2-document corpus, mixing **exact-match** (codes/IDs), **conceptual/paraphrased**, **conflicting-source**, and **unanswerable** questions. Retrieval metrics use local embeddings (free, no API key); generation metrics use a free Gemini Flash judge via RAGAS.

<!-- EVAL_RESULTS_START -->
**Retrieval quality — before / after** (top-k = 5, local bge-large embeddings, real run):

| Mode | Hit@5 | MRR |
|------|------:|----:|
| Dense only (baseline — what most demos ship) | 96.2% | 0.756 |
| Hybrid (dense + BM25, RRF) | 96.2% | 0.772 |
| **Hybrid + rerank (DocuMind)** | **100.0%** | **0.894** |

**Lift from the full pipeline: Hit@5 +3.8 pts, MRR +0.138** vs the dense-only baseline.

**Where the lift comes from** — MRR broken down by question type:

| Question type | Dense | Hybrid+rerank | Δ |
|---------------|------:|--------------:|--:|
| Exact-match (codes/IDs, e.g. `E-4021`, `SKU-9981`) | 0.725 | 0.900 | **+0.175** |
| Conceptual / paraphrased | 0.743 | 0.875 | +0.132 |
| Conflicting sources | 1.000 | 1.000 | +0.000 |

The biggest gain is on **exact-match queries** — exactly where pure dense retrieval struggles, because the embedding of a bare code like `E-4021` sits close to the distractor passage in the *X100* manual ("error code E-4021 does not exist"). BM25 + the cross-encoder pull the correct *X200* passage back to rank 1. This is the whole argument for hybrid retrieval, quantified.

> Generation metrics (RAGAS faithfulness/answer-relevance) and honest-refusal accuracy require a free `GEMINI_API_KEY`; run `python eval/run_eval.py` (without `--no-ragas`) to add them. Results auto-write to [`backend/eval/results.md`](backend/eval/results.md).
<!-- EVAL_RESULTS_END -->

**Multi-turn retrieval** ([`backend/eval/multiturn_dataset.json`](backend/eval/multiturn_dataset.json)) — 10 conversations whose follow-ups are deliberately underspecified ("How do I fix *it*?"). The harness compares retrieving on the raw follow-up vs on the condensed standalone query produced by [`generation/rewrite.py`](backend/app/generation/rewrite.py). Real run (2026-07-04):

| Query used | Hit@5 | MRR |
|------------|------:|----:|
| Raw follow-up | 70.0% | 0.700 |
| **Condensed (DocuMind)** | **90.0%** | **0.900** |

**Lift from condensing: Hit@5 +20 pts, MRR +0.200.** Example: "*And after that window?*" → "*What is AcmeNet's policy on hardware returns after the 30-day window?*" (raw follow-up missed entirely; condensed hit at rank 1). Full per-case table in [`backend/eval/results_multiturn.md`](backend/eval/results_multiturn.md).

**How to reproduce:**
```bash
cd backend
python eval/make_sample_docs.py                       # generate the corpus
python ingest_cli.py eval/sample_docs/*.pdf           # ingest it (Qdrant must be up)
python eval/run_eval.py --no-ragas                    # retrieval metrics (free, fast)
python eval/run_eval.py                               # + RAGAS (needs GEMINI_API_KEY)
python eval/run_multiturn_eval.py                     # follow-up condensing lift (needs LLM key)
```

---

## Decisions & trade-offs

- **Hybrid + rerank over naive top-k.** Pure dense retrieval misses literal tokens (error codes, SKUs, IPs); pure BM25 misses paraphrases. Fusing both and reranking gives the best of each. The eval table above is the receipt. Cost: ~2 extra model loads (both local/free) and a few hundred ms per query — worth it for the accuracy lift.
- **RRF over score-weighted fusion.** Cosine and BM25 scores aren't on the same scale; normalizing them is fiddly and brittle. RRF only needs ranks, so it's robust and parameter-light (one constant `k=60`).
- **Local cross-encoder over Cohere Rerank.** Keeps everything free and offline; `bge-reranker-v2-m3` is competitive and runs on CPU.
- **Chunks never cross page boundaries.** Makes page-level citations exact. We pack by sentence so we don't cut mid-thought, and carry ~15% token overlap into the next chunk so context isn't lost at the seams.
- **~256-token chunks (the CLAUDE spec suggested ~400).** On this corpus, smaller chunks gave more precise citations and finer retrieval granularity, and made `top-k` retrieval actually selective rather than returning most of a tiny index. For large documents, 400+ tokens trades precision for a smaller, cheaper index — `CHUNK_TOKENS` is configurable. This is a deliberate, measured choice, not a default.
- **Honest refusal is enforced in the prompt and verified in eval.** The model must answer only from the numbered sources; unanswerable questions are a scored test case, not an afterthought.
- **Groq primary → Gemini fallback** via the vendored [`shared/llm.py`](backend/shared/llm.py): Groq is fast and free but rate-limited; on a 429 we back off then fall back to Gemini Flash, so the demo stays up.
- **Follow-up condensing over full conversational retrieval.** Multi-turn RAG needs the *retrieval query* to be standalone; we spend one cheap LLM call (`gpt-oss-20b`) rewriting follow-ups instead of embedding whole conversations. First turns skip the call entirely, and any failure falls back to the raw question — the feature can degrade but never break `/ask`.
- **Chat sessions in localStorage, not a backend DB.** DocuMind ships as a standalone zip; SQLite would add schema/migrations/CRUD for zero demo value in a single-user tool. Trade-off: chats don't roam across browsers (noted, accepted).
- **Doc summaries in sidecar JSON, not Qdrant payloads.** Duplicating a summary onto every chunk point bloats payloads, and a synthetic "meta point" would pollute search. One small file per doc, merged into `/documents` at read time.
- **Files stored under `{doc_id}.{ext}`, not the original filename.** Two different files with the same name must not overwrite each other's bytes — the doc_id is content-addressed, so the citation viewer always serves the exact file that was indexed.
- **Snippet highlighting in the PDF viewer is best-effort by design.** pdf.js text items are short line fragments; we match progressively shorter snippet prefixes and fall back to just opening the cited page rather than blocking on fuzzy matching.

## Known limitations

- Text-based documents only (no OCR) — scanned PDFs return an explicit, structured "no extractable text" error surfaced as a toast.
- DOCX/TXT/MD have no physical pages; they're split into ~4K-character pseudo-pages, cited as "part n", and don't open in the PDF viewer (the citation card shows the snippet instead).
- BM25 index is in-memory, rebuilt from Qdrant on first query after an ingest — fine for the free-tier corpus sizes here, would move to a persistent sparse index at scale.
- Chat sessions live in the browser's localStorage (capped at 30 sessions × 200 messages); they don't roam across devices.
- Single-collection store; multi-tenant isolation isn't implemented (out of scope for the portfolio demo).

---

## Run it locally

> **Self-contained.** This folder is everything — the LLM router + tracing are vendored under `backend/shared/`, so DocuMind runs without any files outside `1-documind/`.

**Prerequisites:** Docker (for Qdrant), Python 3.11, Node 20+, and free keys. Copy `.env.example` → `.env` in this folder (Groq + Gemini; Langfuse optional).

```bash
# 1. Vector store (from this 1-documind/ folder)
docker compose up -d                       # Qdrant on :6333

# 2. Backend
cd backend
uv venv .venv && uv pip install -r requirements.txt
./.venv/Scripts/uvicorn app.api.main:app --reload        # http://localhost:8000/docs

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

Upload documents (drag-drop or click) in the left panel, then ask away. Click any `[n]` marker or citation card to open the original PDF at the cited page with the passage highlighted. Ask follow-ups naturally — the condensed search query is visible in each answer's "How this answer was found" inspector. Press `/` to focus the composer; chats persist across reloads (header ▸ Chats).

> **First start takes ~20–60s** while the embedding + reranker models load. The backend preloads them in a background thread on startup (so they're not loaded on your first question), and the UI shows a one-time "warming up" banner driven by `models_ready` on `/health`. Once it disappears, retrieval is fast.

## Deploy (free tier)

- **Backend** → Render / HF Spaces via the [`Dockerfile`](backend/Dockerfile) (self-contained — build from `backend/`: `docker build -t documind-api .`). Use **Qdrant Cloud** free tier for the vector store; set `QDRANT_URL`/`QDRANT_API_KEY`.
- **Frontend** → Vercel. Set `VITE_API_BASE` to the deployed backend URL.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ingest` | Upload + index documents (PDF/DOCX/TXT/MD) |
| GET | `/documents` | List indexed docs (incl. summary + suggested questions) |
| GET | `/documents/{id}/file` | Serve the original file (citation PDF viewer) |
| DELETE | `/documents/{id}` | Remove a doc (vectors + stored file + sidecar meta) |
| POST | `/ask` | Grounded answer + citations (JSON); accepts `history`, `include_trace` |
| POST | `/ask/stream` | Same, streamed via SSE (meta → sources → trace → tokens → citations) |
| GET | `/health` | Qdrant status + indexed chunk count + model readiness |
