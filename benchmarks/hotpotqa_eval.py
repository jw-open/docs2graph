"""
HotpotQA benchmark evaluation for doc2graph.

Measures document recall: given a natural language question and a set of
candidate Wikipedia articles, does doc2graph surface the correct supporting
documents in its top-k retrieved nodes?

HotpotQA is a multi-hop QA dataset — each question requires reasoning over
exactly 2 supporting documents from a distractor pool of ~10 articles.
This makes it a strong test of doc2graph's relevance ranking.

Usage
-----
1. Download HotpotQA:
   https://hotpotqa.github.io/
   Get hotpot_dev_distractor_v1.json (or fullwiki version)

2. Run:
   python benchmarks/hotpotqa_eval.py --data hotpot_dev_distractor_v1.json --k 5

What this measures
------------------
Supporting document recall@k:
  recall@k = |gold_titles ∩ retrieved_node_labels| / |gold_titles|
  mean_recall@k = average across all questions

HotpotQA gold always has exactly 2 supporting titles, so:
  - recall = 1.0 means both supporting docs retrieved
  - recall = 0.5 means one of two retrieved
  - recall = 0.0 means neither retrieved

Baseline: random k from ~10 = ~2k/10. At k=5: ~1.0 expected by chance.
Our goal: high recall at small k (k=2 or k=3) to reduce token usage.

Metric definition
-----------------
  perfect_recall@k: fraction of questions where both docs are retrieved
  mean_recall@k:    average fraction of gold docs retrieved
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from doc2graph import DocumentGraph


def evaluate(
    data_path: Path,
    k: int = 5,
    alpha: float = 0.85,
    limit: int = 0,
) -> Dict:
    """
    Run doc2graph recall evaluation on HotpotQA distractor set.

    Parameters
    ----------
    data_path : Path
        Path to hotpot_dev_distractor_v1.json (or train).
    k : int
        Number of top-k nodes for doc2graph.rank().
    alpha : float
        PPR damping factor.
    limit : int
        If > 0, evaluate only the first N questions.

    Returns
    -------
    dict
        Evaluation results with mean_recall and perfect_recall fraction.
    """
    print(f"Loading {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    print(f"Evaluating {len(data)} questions (k={k}, alpha={alpha})...")

    recalls: List[float] = []

    for item in data:
        question: str = item["question"]

        # Gold supporting document titles
        gold_titles = {title.lower() for title, _ in item["supporting_facts"]}

        # Context: list of [title, [sentence, sentence, ...]]
        # Build a DocumentGraph from the candidate articles
        texts = [
            {
                "title": title,
                "content": " ".join(sentences),
            }
            for title, sentences in item["context"]
        ]
        g = DocumentGraph.from_texts(texts)

        result = g.rank(question, k=k, alpha=alpha)

        retrieved_labels = {n["label"].lower() for n in result["nodes"]}
        hits = len(gold_titles & retrieved_labels)
        recall = hits / len(gold_titles) if gold_titles else 0.0
        recalls.append(recall)

    if not recalls:
        print("No questions scored.")
        return {}

    mean_recall = sum(recalls) / len(recalls)
    perfect = sum(1 for r in recalls if r == 1.0) / len(recalls)
    partial = sum(1 for r in recalls if 0 < r < 1.0) / len(recalls)
    zero = sum(1 for r in recalls if r == 0.0) / len(recalls)

    return {
        "dataset": "HotpotQA (distractor)",
        "k": k,
        "alpha": alpha,
        "total_questions": len(recalls),
        "mean_recall": round(mean_recall, 4),
        "perfect_recall_fraction": round(perfect, 4),
        "partial_recall_fraction": round(partial, 4),
        "zero_recall_fraction": round(zero, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="doc2graph HotpotQA benchmark")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to hotpot_dev_distractor_v1.json",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k nodes (default: 5)")
    parser.add_argument("--alpha", type=float, default=0.85, help="PPR damping (default: 0.85)")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N questions (0 = all)")
    args = parser.parse_args()

    results = evaluate(
        data_path=args.data,
        k=args.k,
        alpha=args.alpha,
        limit=args.limit,
    )

    print("\n=== doc2graph HotpotQA Evaluation Results ===")
    for key, val in results.items():
        print(f"  {key:35s}: {val}")


if __name__ == "__main__":
    main()
