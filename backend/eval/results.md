# DocuMind — Evaluation Results

## Retrieval quality (before / after hybrid + rerank)

30-question eval set · top-k = 5 · local bge-large embeddings.

| Mode | Hit@5 | MRR |
|------|------:|----:|
| Dense only (baseline) | 96.2% | 0.756 |
| Hybrid (dense+BM25, RRF) | 100.0% | 0.791 |
| Hybrid + rerank (DocuMind) | 100.0% | 0.894 |

**Lift from the full pipeline:** Hit@5 +3.8 pts, MRR +0.138 vs the dense-only baseline.

### MRR by question type (where the lift comes from)

| Question type | Dense | Hybrid+rerank | Δ |
|---------------|------:|--------------:|--:|
| Conceptual / paraphrased | 0.743 | 0.875 | +0.132 |
| Conflicting sources | 1.000 | 1.000 | +0.000 |
| Exact-match (codes/IDs) | 0.725 | 0.900 | +0.175 |
