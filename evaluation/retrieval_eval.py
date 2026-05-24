"""
Retrieval performance evaluator.

Loads evaluation/gold_retrieval.jsonl and scores three retrieval pipelines
side-by-side on the same queries:

  1. dense          — FAISS vector search alone
  2. hybrid         — FAISS + BM25 merged with Reciprocal Rank Fusion
  3. hybrid_rerank  — hybrid + cross-encoder rerank

Metrics (computed for k ∈ {1, 3, 5, 10}):
  - Hit@k         : did any relevant chunk make the top-k?
  - MRR           : mean reciprocal rank of the first relevant hit (0 if miss)
  - Recall@k      : |relevant ∩ top-k| / |relevant|
  - Precision@k   : |relevant ∩ top-k| / k
  - nDCG@k        : DCG@k / IDCG@k  (binary relevance)

Run:
    python -m evaluation.retrieval_eval
    python -m evaluation.retrieval_eval --gold evaluation/gold_retrieval.jsonl --k 1 3 5 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
from retrieval.vector_store import vector_store
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_GOLD = Path("evaluation/gold_retrieval.jsonl")
DEFAULT_OUT = Path("evaluation/retrieval_eval_results.json")
DEFAULT_KS = [1, 3, 5, 10]

# Search only over leaf chunks (the things gold IDs actually point at)
CHILD_FILTER = {"chunk_type": {"$ne": "parent"}}


# ── Stage 1/2/3 runners ─────────────────────────────────────────────────────

def _dense_search(query: str, k: int) -> list[dict]:
    return vector_store.search(query=query, top_k=k, where=CHILD_FILTER)


def _hybrid_search(query: str, k: int) -> list[dict]:
    return hybrid_search(query=query, top_k=k, where=CHILD_FILTER)


def _hybrid_rerank(query: str, k: int) -> list[dict]:
    # Over-fetch then rerank down to k
    pool = hybrid_search(query=query, top_k=max(k * 4, settings.retrieval_top_k), where=CHILD_FILTER)
    return rerank(query=query, results=pool, top_k=k)


# ── Metric primitives ───────────────────────────────────────────────────────

def _retrieved_ids(results: list[dict]) -> list[str]:
    return [r.get("id", "") for r in results]


def _hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(rid in relevant for rid in retrieved[:k]) else 0.0


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG: all relevant chunks ranked at the top, up to k
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ── Per-query scoring ───────────────────────────────────────────────────────

def _score_one(retrieved_ids: list[str], relevant: set[str], ks: list[int]) -> dict:
    scores = {"mrr": _reciprocal_rank(retrieved_ids, relevant)}
    for k in ks:
        scores[f"hit@{k}"] = _hit_at_k(retrieved_ids, relevant, k)
        scores[f"recall@{k}"] = _recall_at_k(retrieved_ids, relevant, k)
        scores[f"precision@{k}"] = _precision_at_k(retrieved_ids, relevant, k)
        scores[f"ndcg@{k}"] = _ndcg_at_k(retrieved_ids, relevant, k)
    return scores


def _aggregate(per_query: list[dict]) -> dict:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}


# ── Main eval loop ──────────────────────────────────────────────────────────

STAGES = {
    "dense":         _dense_search,
    "hybrid":        _hybrid_search,
    "hybrid_rerank": _hybrid_rerank,
}


def evaluate_retrieval(
    gold_path: Path = DEFAULT_GOLD,
    ks: list[int] = DEFAULT_KS,
    out_path: Path = DEFAULT_OUT,
) -> dict:
    if not gold_path.exists():
        raise FileNotFoundError(
            f"Gold dataset not found at {gold_path}. "
            "Build it first: python -m evaluation.build_gold_dataset --n 50"
        )

    with open(gold_path, "r", encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]

    if not gold:
        raise ValueError(f"Gold dataset {gold_path} is empty")

    max_k = max(ks)
    logger.info("Evaluating %d queries over stages: %s (k up to %d)",
                len(gold), list(STAGES.keys()), max_k)

    per_stage_per_query: dict[str, list[dict]] = {s: [] for s in STAGES}
    per_stage_timings: dict[str, list[float]] = {s: [] for s in STAGES}
    failures: list[dict] = []

    pbar = tqdm(
        gold,
        desc="Retrieval eval",
        unit="q",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    for i, item in enumerate(pbar, start=1):
        query = item["question"]
        relevant = set(item.get("relevant_chunk_ids", []))
        if not relevant:
            pbar.set_postfix_str(f"qid={item.get('qid','?')[:24]} skipped", refresh=False)
            continue

        for stage_name, fn in STAGES.items():
            t0 = time.time()
            try:
                results = fn(query, max_k)
                retrieved = _retrieved_ids(results)
            except Exception as e:
                logger.warning("[%d/%d] %s failed for qid=%s: %s",
                               i, len(gold), stage_name, item.get("qid", "?"), e)
                failures.append({"qid": item.get("qid"), "stage": stage_name, "error": str(e)})
                retrieved = []
            elapsed_ms = (time.time() - t0) * 1000

            per_stage_per_query[stage_name].append({
                "qid": item.get("qid", f"q_{i}"),
                "scores": _score_one(retrieved, relevant, ks),
                "elapsed_ms": elapsed_ms,
                "first_hit_rank": next(
                    (idx + 1 for idx, rid in enumerate(retrieved) if rid in relevant),
                    None,
                ),
            })
            per_stage_timings[stage_name].append(elapsed_ms)

        pbar.set_postfix_str(f"qid={item.get('qid','?')[:24]}", refresh=False)
    pbar.close()

    aggregates = {
        stage: _aggregate([q["scores"] for q in per_stage_per_query[stage]])
        for stage in STAGES
    }
    avg_timings = {
        stage: round(sum(t) / max(len(t), 1), 1)
        for stage, t in per_stage_timings.items()
    }

    summary = {
        "gold_path": str(gold_path),
        "num_queries": len(gold),
        "ks": ks,
        "stages": list(STAGES.keys()),
        "aggregate_scores": aggregates,
        "avg_elapsed_ms": avg_timings,
        "failures": failures,
        "per_query": per_stage_per_query,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    _print_report(aggregates, avg_timings, ks, len(gold), out_path)
    return summary


def _print_report(
    aggregates: dict,
    avg_timings: dict,
    ks: list[int],
    n_queries: int,
    out_path: Path,
) -> None:
    print()
    print("=" * 78)
    print(f"RETRIEVAL EVAL — {n_queries} queries")
    print("=" * 78)

    metric_order = ["mrr"] + [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] \
        + [f"precision@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]

    header = f"{'metric':<14} | " + " | ".join(f"{s:>14}" for s in aggregates.keys())
    print(header)
    print("-" * len(header))

    for metric in metric_order:
        row = f"{metric:<14} | " + " | ".join(
            f"{aggregates[s].get(metric, 0.0):>14.3f}" for s in aggregates
        )
        print(row)

    print("-" * len(header))
    timing_row = f"{'avg_ms':<14} | " + " | ".join(f"{avg_timings[s]:>14.1f}" for s in aggregates)
    print(timing_row)
    print("=" * 78)
    print(f"Full per-query results written to: {out_path}")
    print()


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Gold dataset JSONL path")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write results JSON")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS, help="Cutoffs (default: 1 3 5 10)")
    args = parser.parse_args()

    setup_logging(level="INFO")
    evaluate_retrieval(gold_path=args.gold, ks=args.k, out_path=args.out)


if __name__ == "__main__":
    _cli()
