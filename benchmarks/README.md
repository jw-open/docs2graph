# Benchmarks

## HotpotQA — Supporting Document Recall

Measures whether doc2graph surfaces the correct supporting documents for a multi-hop question.

HotpotQA is a strong benchmark because:
- Each question requires **exactly 2 supporting documents** from a pool of ~10 distractors
- Multi-hop: the answer requires reasoning across both docs
- Standard — used by RAG papers to compare retrieval quality

**Metric:** Supporting Document Recall@k
- `recall = 1.0` → both supporting docs retrieved
- `recall = 0.5` → one of two retrieved
- `recall = 0.0` → neither retrieved

| k | Random baseline | doc2graph target |
|---|---|---|
| 2 | ~36% perfect | ≥ 55% perfect |
| 3 | ~58% perfect | ≥ 70% perfect |
| 5 | ~83% perfect | ≥ 85% perfect |

### Setup

```bash
# 1. Download HotpotQA distractor dev set
#    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# 2. Install dev deps
pip install -e ".[dev]"

# 3. Run eval (k=5)
python benchmarks/hotpotqa_eval.py --data hotpot_dev_distractor_v1.json --k 5

# Quick smoke test (first 100 questions)
python benchmarks/hotpotqa_eval.py --data hotpot_dev_distractor_v1.json --k 5 --limit 100
```

### Expected output

```
=== doc2graph HotpotQA Evaluation Results ===
  dataset                            : HotpotQA (distractor)
  k                                  : 5
  alpha                              : 0.85
  total_questions                    : 7405
  mean_recall                        : 0.XXXX
  perfect_recall_fraction            : 0.XXXX
  partial_recall_fraction            : 0.XXXX
  zero_recall_fraction               : 0.XXXX
```

---

## Planned benchmarks

| Benchmark | Type | Status |
|---|---|---|
| HotpotQA | Multi-hop QA, 2 supporting docs | ✅ eval script ready |
| MuSiQue | Multi-hop, harder 2-4 hops | planned v0.2.0 |
| 2WikiMultihopQA | Cross-document reasoning | planned v0.2.0 |
