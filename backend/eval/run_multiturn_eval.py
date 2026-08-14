"""Multi-turn retrieval eval: raw follow-up query vs condensed standalone query.

Follow-up questions ("How do I fix it?") retrieve poorly because their key
terms live in earlier turns. DocuMind condenses follow-ups with one cheap LLM
call before retrieval (app/generation/rewrite.py). This harness quantifies
that lift: Hit@5 / MRR retrieving on the RAW follow-up vs on the CONDENSED
query, over 10 short conversations.

Needs: Qdrant running with the sample corpus ingested (see run_eval.py
docstring) and a GROQ_API_KEY/GEMINI_API_KEY for the rewrite call.

Usage:
    python eval/run_multiturn_eval.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.generation.rewrite import condense_question  # noqa: E402
from app.models import ChatTurn  # noqa: E402
from app.retrieval.pipeline import retrieve  # noqa: E402

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "multiturn_dataset.json"
RESULTS = EVAL_DIR / "results_multiturn.md"
TOP_K = 5


def _first_hit_rank(chunks, gold: str) -> int | None:
    gold = gold.lower().strip()
    for i, rc in enumerate(chunks, start=1):
        if gold and gold in rc.chunk.text.lower():
            return i
    return None


def main() -> int:
    # Windows consoles default to cp1252, which can't print "«»" etc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases = json.loads(DATASET.read_text(encoding="utf-8"))["cases"]
    print(f"Loaded {len(cases)} multi-turn cases.")

    rows = []
    for case in cases:
        history = [ChatTurn(**t) for t in case["history"]]
        raw_q = case["followup"]
        condensed_q, rewritten = condense_question(raw_q, history)

        raw_rank = _first_hit_rank(
            retrieve(raw_q, mode="hybrid_rerank", top_k=TOP_K), case["gold_substring"]
        )
        cond_rank = _first_hit_rank(
            retrieve(condensed_q, mode="hybrid_rerank", top_k=TOP_K),
            case["gold_substring"],
        )
        rows.append(
            {
                "id": case["id"],
                "raw": raw_q,
                "condensed": condensed_q,
                "rewritten": rewritten,
                "raw_rr": 1.0 / raw_rank if raw_rank else 0.0,
                "cond_rr": 1.0 / cond_rank if cond_rank else 0.0,
            }
        )
        print(
            f"  {case['id']}: raw_rr={rows[-1]['raw_rr']:.2f} "
            f"cond_rr={rows[-1]['cond_rr']:.2f}  «{condensed_q}»"
        )

    raw_mrr = statistics.mean(r["raw_rr"] for r in rows)
    cond_mrr = statistics.mean(r["cond_rr"] for r in rows)
    raw_hit = sum(r["raw_rr"] > 0 for r in rows) / len(rows)
    cond_hit = sum(r["cond_rr"] > 0 for r in rows) / len(rows)

    lines = [
        "# DocuMind — Multi-turn Retrieval Eval",
        "",
        f"{len(rows)} conversations · top-k = {TOP_K} · full hybrid+rerank pipeline.",
        "Raw = retrieving on the literal follow-up; Condensed = retrieving on the",
        "standalone query produced by the follow-up rewriting step.",
        "",
        "| Query used | Hit@5 | MRR |",
        "|------------|------:|----:|",
        f"| Raw follow-up | {raw_hit*100:.1f}% | {raw_mrr:.3f} |",
        f"| Condensed (DocuMind) | {cond_hit*100:.1f}% | {cond_mrr:.3f} |",
        "",
        f"**Lift from condensing:** Hit@5 {(cond_hit-raw_hit)*100:+.1f} pts, "
        f"MRR {cond_mrr-raw_mrr:+.3f}.",
        "",
        "## Example rewrites",
        "",
        "| Follow-up | Condensed query |",
        "|-----------|-----------------|",
    ]
    for r in rows:
        lines.append(f"| {r['raw']} | {r['condensed']} |")
    lines.append("")

    RESULTS.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS}")
    print(f"Raw MRR {raw_mrr:.3f} → Condensed MRR {cond_mrr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
