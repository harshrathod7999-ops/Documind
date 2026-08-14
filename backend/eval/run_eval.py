"""DocuMind evaluation harness.

Two layers, deliberately separated by cost:

1. RETRIEVAL metrics (free, no API key, local embeddings) — run by default.
   For each question we compare three retrieval modes:
     * dense          (baseline: dense vectors only)
     * hybrid         (dense + BM25 fused with RRF)
     * hybrid_rerank  (full pipeline: fusion + cross-encoder rerank)
   Metrics:
     * Hit@5 / Recall : did a retrieved chunk contain the gold answer span?
     * MRR            : reciprocal rank of the first chunk containing it.
   This produces the BEFORE/AFTER lift table proving the hybrid+rerank design.

2. GENERATION metrics via RAGAS (faithfulness, answer relevance) using a FREE
   Gemini Flash judge. Requires GEMINI_API_KEY; skipped with --no-ragas or if
   the key is absent. Also scores honest-refusal accuracy on unanswerable Qs.

Usage:
    # one-time corpus setup (needs Qdrant running: docker compose up -d)
    python eval/make_sample_docs.py
    python ingest_cli.py eval/sample_docs/atlas_x200_manual.pdf eval/sample_docs/acmenet_support_policy.pdf

    python eval/run_eval.py                 # retrieval + RAGAS (if key present)
    python eval/run_eval.py --no-ragas      # retrieval only (fast, free)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Make `app` importable when run from the backend dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.pipeline import retrieve  # noqa: E402

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "dataset.json"
RESULTS = EVAL_DIR / "results.md"
MODES = ["dense", "hybrid", "hybrid_rerank"]
TOP_K = 5


def load_cases() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]


def _first_hit_rank(chunks, gold: str) -> int | None:
    """1-based rank of the first chunk containing the gold span, else None."""
    gold = gold.lower().strip()
    for i, rc in enumerate(chunks, start=1):
        if gold and gold in rc.chunk.text.lower():
            return i
    return None


def run_retrieval_eval(cases: list[dict]) -> dict:
    """Compute Hit@5 and MRR per mode over the answerable questions.

    Also breaks MRR down by question type, since the value of hybrid + rerank
    is concentrated in exact-match queries (codes/IDs) where dense embeddings
    rank near-duplicate distractors highly.
    """
    answerable = [c for c in cases if not c.get("unanswerable")]
    per_mode: dict[str, dict] = {}
    types = sorted({c["type"] for c in answerable})

    for mode in MODES:
        hits, rrs = 0, []
        by_type: dict[str, list[float]] = {t: [] for t in types}
        for case in answerable:
            chunks = retrieve(case["question"], mode=mode, top_k=TOP_K)
            rank = _first_hit_rank(chunks, case["gold_substring"])
            rr = (1.0 / rank) if rank is not None else 0.0
            rrs.append(rr)
            by_type[case["type"]].append(rr)
            if rank is not None:
                hits += 1
        n = len(answerable)
        per_mode[mode] = {
            "hit_at_5": hits / n,
            "mrr": statistics.mean(rrs) if rrs else 0.0,
            "n": n,
            "mrr_by_type": {t: statistics.mean(v) for t, v in by_type.items() if v},
        }
    return per_mode


def run_refusal_eval(cases: list[dict]) -> dict:
    """Check the full RAG answers honest-refuse on unanswerable questions.

    Needs an LLM (uses the shared router). Skipped if generation isn't wired.
    """
    from app.generation.answer import _NOT_FOUND, generate_answer

    unanswerable = [c for c in cases if c.get("unanswerable")]
    correct = 0
    details = []
    for case in unanswerable:
        chunks = retrieve(case["question"], mode="hybrid_rerank", top_k=TOP_K)
        resp = generate_answer(case["question"], chunks)
        refused = (not resp.grounded) or (_NOT_FOUND.lower() in resp.answer.lower())
        correct += int(refused)
        details.append({"id": case["id"], "refused": refused, "answer": resp.answer[:120]})
    return {
        "refusal_accuracy": correct / len(unanswerable) if unanswerable else 0.0,
        "n": len(unanswerable),
        "details": details,
    }


def run_ragas_eval(cases: list[dict]) -> dict | None:
    """RAGAS faithfulness + answer relevance with a free Gemini Flash judge."""
    import os

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("• Skipping RAGAS (no GEMINI_API_KEY/GOOGLE_API_KEY set).")
        return None
    try:
        from datasets import Dataset
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception as e:  # pragma: no cover
        print(f"• Skipping RAGAS (deps not installed: {e}). See requirements-eval.txt")
        return None

    from app.generation.answer import generate_answer

    answerable = [c for c in cases if not c.get("unanswerable")]
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for case in answerable:
        chunks = retrieve(case["question"], mode="hybrid_rerank", top_k=TOP_K)
        resp = generate_answer(case["question"], chunks)
        rows["question"].append(case["question"])
        rows["answer"].append(resp.answer)
        rows["contexts"].append([rc.chunk.text for rc in chunks])
        rows["ground_truth"].append(case["expected_answer"])

    judge = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    result = evaluate(
        Dataset.from_dict(rows),
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=emb,
    )
    # EvaluationResult isn't reliably dict-like across ragas versions; derive the
    # aggregate scores from the per-sample dataframe, which is stable.
    df = result.to_pandas()
    metric_cols = [c for c in ("faithfulness", "answer_relevancy") if c in df.columns]
    return {c: float(df[c].mean()) for c in metric_cols}


def write_report(retrieval: dict, refusal: dict | None, ragas: dict | None) -> None:
    lines = ["# DocuMind — Evaluation Results", ""]
    lines += [
        "## Retrieval quality (before / after hybrid + rerank)",
        "",
        f"30-question eval set · top-k = {TOP_K} · local bge-large embeddings.",
        "",
        "| Mode | Hit@5 | MRR |",
        "|------|------:|----:|",
    ]
    label = {
        "dense": "Dense only (baseline)",
        "hybrid": "Hybrid (dense+BM25, RRF)",
        "hybrid_rerank": "Hybrid + rerank (DocuMind)",
    }
    for mode in MODES:
        m = retrieval[mode]
        lines.append(f"| {label[mode]} | {m['hit_at_5']*100:.1f}% | {m['mrr']:.3f} |")

    base = retrieval["dense"]
    full = retrieval["hybrid_rerank"]
    lift_hit = (full["hit_at_5"] - base["hit_at_5"]) * 100
    lift_mrr = full["mrr"] - base["mrr"]
    lines += [
        "",
        f"**Lift from the full pipeline:** Hit@5 {lift_hit:+.1f} pts, "
        f"MRR {lift_mrr:+.3f} vs the dense-only baseline.",
        "",
        "### MRR by question type (where the lift comes from)",
        "",
        "| Question type | Dense | Hybrid+rerank | Δ |",
        "|---------------|------:|--------------:|--:|",
    ]
    type_label = {
        "exact_match": "Exact-match (codes/IDs)",
        "conceptual": "Conceptual / paraphrased",
        "conflict": "Conflicting sources",
    }
    for t in sorted(base["mrr_by_type"]):
        b = base["mrr_by_type"][t]
        f = full["mrr_by_type"].get(t, 0.0)
        lines.append(f"| {type_label.get(t, t)} | {b:.3f} | {f:.3f} | {f-b:+.3f} |")
    lines.append("")

    if refusal is not None:
        lines += [
            "## Honest refusal (unanswerable questions)",
            "",
            f"Refusal accuracy on {refusal['n']} out-of-corpus questions: "
            f"**{refusal['refusal_accuracy']*100:.1f}%** "
            "(higher = the model correctly says “not in the documents”).",
            "",
        ]

    if ragas is not None:
        lines += ["## Generation quality (RAGAS, Gemini Flash judge)", ""]
        for k, v in ragas.items():
            lines.append(f"- **{k}**: {v:.3f}")
        lines.append("")

    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS}")
    print("\n".join(lines))


def main() -> int:
    # Windows consoles default to cp1252, which can't print "Δ" etc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ragas", action="store_true", help="skip LLM-judged metrics")
    ap.add_argument("--no-refusal", action="store_true", help="skip refusal eval (LLM)")
    args = ap.parse_args()

    cases = load_cases()
    print(f"Loaded {len(cases)} eval cases.\nRunning retrieval eval across {MODES}…")
    retrieval = run_retrieval_eval(cases)

    refusal = None if args.no_refusal else _safe(run_refusal_eval, cases, "refusal")
    ragas = None if args.no_ragas else _safe(run_ragas_eval, cases, "RAGAS")

    write_report(retrieval, refusal, ragas)
    return 0


def _safe(fn, cases, name):
    try:
        return fn(cases)
    except Exception as e:
        print(f"• {name} eval skipped ({type(e).__name__}: {e}).")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
