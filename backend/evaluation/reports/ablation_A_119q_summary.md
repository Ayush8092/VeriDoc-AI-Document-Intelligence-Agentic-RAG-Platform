# Ablation A — Partial Results (119 Questions)

> **This report is based on 119 COMPLETED questions only. It is not the full benchmark. No result for an incomplete question was estimated, interpolated, or fabricated — every number below is computed directly and only from the questions that actually finished.**

- Checkpoint source: `evaluation\reports\ablation_A.checkpoint.json`
- Completed: **119** of 300 planned
- Generated: 2026-09-01T16:10:45.476248+00:00

## Application / grounding metrics

| Metric | Value |
| --- | --- |
| n_questions | 119 |
| grounded_answer_rate | N/A |
| refusal_accuracy | 0.9160 |
| correct_answer_rate | 0.8113 |
| citation_precision | N/A |
| citation_recall | 0.8302 |
| claim_grounding_rate | N/A |
| unsupported_claim_rate | N/A |
| hallucination_rate | N/A |
| answerable_sensitivity | 0.9340 |
| unanswerable_specificity | 0.7692 |

## Retrieval metrics

| Metric | Value |
| --- | --- |
| recall@1 | 0.6973 |
| precision@1 | 0.7736 |
| hit_rate@1 | 0.7736 |
| ndcg@1 | 0.7736 |
| recall@3 | 0.8813 |
| precision@3 | 0.3427 |
| hit_rate@3 | 0.9245 |
| ndcg@3 | 0.8334 |
| recall@5 | 0.9151 |
| precision@5 | 0.2208 |
| hit_rate@5 | 0.9245 |
| ndcg@5 | 0.8488 |
| recall@10 | 0.9198 |
| precision@10 | 0.1113 |
| hit_rate@10 | 0.9245 |
| ndcg@10 | 0.8507 |
| mrr | 0.8428 |

## What's NOT in this report

- Any question beyond the 119 completed here — not run yet, not estimated.
- RAGAS/TruLens scores, unless the checkpoint's records already carried them (this script only computes what `app.evaluation.metrics`/`retrieval_metrics` can derive from the fields present).
- Anything requiring a live API call — this script made none.

Re-run this script against an updated checkpoint (more completed questions) at any time to regenerate all three files with more data — nothing here needs to be hand-edited.
