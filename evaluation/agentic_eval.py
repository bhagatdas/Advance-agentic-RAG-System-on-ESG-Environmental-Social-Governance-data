"""
Agentic RAG end-to-end evaluator.

Loads QA pairs from JSON files in QNA_data/ (or a single file), runs each
question through the full LangGraph workflow via invoke_query(), and scores
the agent's answer against the gold answer using four complementary metrics:

  - correctness        : LLM-as-judge verdict (correct / partial / incorrect)
                         translated to a continuous [0, 1] score. The primary
                         signal — captures semantic equivalence even when
                         wording differs.
  - similarity         : cosine similarity between embeddings of the predicted
                         answer and the gold answer. Cheap secondary signal;
                         high-similarity refusals stay flagged as incorrect by
                         the judge.
  - citation_coverage  : produced by the citation verifier inside Reasoning.
                         Fraction of cited (doc, page) pairs that actually
                         exist in the retrieved chunks.
  - faithfulness       : produced by the faithfulness checker inside Reasoning.
                         Claim-level groundedness against retrieved chunks.

A composite score is a weighted blend; the per-query pass flag fires when
composite >= --pass-threshold (default 0.65). Agentic signals are recorded too
(sub-question count, agents invoked, validation retries, latency) for
diagnostics but do not affect the pass flag.

Input formats supported (both keyed under "qa_pairs"):
  - honeywell_esg_qa.json   — {"id": int, "question": str, "answer": str}
  - honeywell_qa_pairs.json — {"question": str, "answer": str} (dedup by
                              stripping "(Extended #N)" suffix)

Run:
    python -m evaluation.agentic_eval                              # all files in QNA_data/
    python -m evaluation.agentic_eval --dataset QNA_data/honeywell_esg_qa.json
    python -m evaluation.agentic_eval --limit 20                   # smoke test
    python -m evaluation.agentic_eval --pass-threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from graph.workflow import invoke_query
from utils.embeddings import embed_texts
from utils.llm import get_structured_llm
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("QNA_data")
DEFAULT_OUT = Path("evaluation/agentic_eval_results.json")
DEFAULT_PASS_THRESHOLD = 0.65

# Composite score weights — must sum to 1.0
WEIGHTS = {
    "correctness": 0.45,
    "similarity": 0.20,
    "citation_coverage": 0.15,
    "faithfulness": 0.20,
}

# Verdict → numeric score mapping for the LLM judge
VERDICT_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

# Strip "(Extended #N)" trailing tags introduced by the duplicated dataset
_EXTENDED_SUFFIX = re.compile(r"\s*\(Extended\s+#\d+\)\s*$", re.IGNORECASE)

# Refusal markers — kept in sync with agents/reasoning.py & adversarial_eval.py
_REFUSAL_MARKERS = [
    "i can only answer questions about the esg reports",
    "outside the scope of the available documents",
    "this specific information is not available",
    "the documents do not contain",
    "not mentioned in the report",
    "not available in the uploaded",
    "is not in the provided context",
]


# ── Judge schema ────────────────────────────────────────────────────────────

class JudgeVerdict(BaseModel):
    """Structured verdict from the LLM-as-judge."""

    verdict: Literal["correct", "partial", "incorrect"] = Field(
        description=(
            "'correct' if the predicted answer fully captures the key facts "
            "in the gold answer (paraphrase OK); 'partial' if it captures "
            "some but is missing or hedges on key facts; 'incorrect' if it "
            "contradicts, refuses, or misses the substance entirely."
        )
    )
    reasoning: str = Field(
        description="One or two sentences justifying the verdict — name the key fact(s) that matched or were missing."
    )


_JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating answers from a Retrieval-Augmented Generation system about Honeywell's ESG reports.

You will be given a QUESTION, the GOLD ANSWER (ground truth from a human-curated dataset), and the PREDICTED ANSWER (from the system).

Judge whether the predicted answer conveys the same substantive facts as the gold answer. Be strict about factual content (numbers, names, dates, percentages) but lenient about phrasing — paraphrases and reorderings are fine. Extra correct context in the prediction is fine; missing key facts is not.

If the predicted answer refuses ("I cannot answer", "not in the documents", etc.) but the gold answer contains real information, that is INCORRECT.

Return a structured verdict."""


# ── Loader ──────────────────────────────────────────────────────────────────

def _normalize_question(q: str) -> str:
    return _EXTENDED_SUFFIX.sub("", q or "").strip().lower()


def _load_qa_file(path: Path) -> list[dict]:
    """Read one QA JSON file, normalize, and dedupe by question."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("qa_pairs", []) if isinstance(data, dict) else data
    seen: set[str] = set()
    items: list[dict] = []
    for i, row in enumerate(raw):
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        if not q or not a:
            continue

        # Strip "(Extended #N)" from the visible question too, for cleaner logs
        q_clean = _EXTENDED_SUFFIX.sub("", q).strip()
        key = _normalize_question(q_clean)
        if key in seen:
            continue
        seen.add(key)

        items.append({
            "qid": f"{path.stem}::{row.get('id', i + 1)}",
            "source": path.name,
            "question": q_clean,
            "gold_answer": a,
        })
    return items


def _load_dataset(dataset: Optional[Path], data_dir: Path) -> list[dict]:
    """Either load one file or every *.json in the data dir."""
    if dataset is not None:
        return _load_qa_file(dataset)

    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON QA files found in {data_dir}")

    all_items: list[dict] = []
    for f in files:
        all_items.extend(_load_qa_file(f))
    return all_items


# ── Metric primitives ───────────────────────────────────────────────────────

def _is_refusal(answer: str) -> bool:
    if not answer:
        return True
    low = answer.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def _cosine_similarity(a: str, b: str) -> float:
    try:
        emb = embed_texts([a, b])
        v1, v2 = np.asarray(emb[0]), np.asarray(emb[1])
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
    except Exception as e:
        logger.warning("Embedding similarity failed: %s", e)
        return 0.0


def _llm_judge(question: str, gold: str, predicted: str) -> JudgeVerdict:
    """Single structured LLM call — light model is plenty for this task."""
    judge = get_structured_llm(JudgeVerdict, task_type="light", temperature=0.0)
    prompt = (
        f"{_JUDGE_SYSTEM_PROMPT}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER:\n{gold}\n\n"
        f"PREDICTED ANSWER:\n{predicted}\n\n"
        f"Return your verdict."
    )
    return judge.invoke(prompt)


def _composite(scores: dict) -> float:
    return round(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 4)


# ── Agents-used detection ───────────────────────────────────────────────────

def _agents_used(trace: list[dict]) -> list[str]:
    """Pull the unique agent names that fired from the execution trace."""
    seen: list[str] = []
    for step in trace or []:
        name = step.get("agent") or step.get("node") or step.get("name") or ""
        if name and name not in seen:
            seen.append(name)
    return seen


# ── Per-query scoring ───────────────────────────────────────────────────────

def _score_one(item: dict, result: dict, pass_threshold: float) -> dict:
    """Score a single QA pair against the agent's response."""
    predicted = (result.get("answer") or "").strip()
    gold = item["gold_answer"]
    question = item["question"]

    refused = _is_refusal(predicted)

    # 1. LLM-as-judge correctness — primary signal
    try:
        verdict = _llm_judge(question, gold, predicted)
        correctness = VERDICT_SCORES[verdict.verdict]
        judge_verdict = verdict.verdict
        judge_reason = verdict.reasoning
    except Exception as e:
        logger.warning("Judge failed for %s: %s", item.get("qid"), e)
        correctness = 0.0
        judge_verdict = "error"
        judge_reason = f"judge error: {e}"

    # 2. Semantic similarity — cheap secondary signal
    similarity = _cosine_similarity(predicted, gold) if predicted else 0.0
    similarity = max(0.0, similarity)  # cosine can be slightly negative — clip

    # 3 & 4. Pulled straight from the workflow's own self-checks
    citation_coverage = float(result.get("citation_coverage", 0.0) or 0.0)
    faithfulness = float(result.get("faithfulness_score", 0.0) or 0.0)

    scores = {
        "correctness": correctness,
        "similarity": similarity,
        "citation_coverage": citation_coverage,
        "faithfulness": faithfulness,
    }
    composite = _composite(scores)

    record = {
        "qid": item["qid"],
        "source": item["source"],
        "question": question,
        "gold_preview": gold[:200],
        "predicted_preview": predicted[:200],
        "refused": refused,
        "judge_verdict": judge_verdict,
        "judge_reasoning": judge_reason,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "composite_score": composite,
        "passed": composite >= pass_threshold and judge_verdict != "incorrect",
        # Agentic diagnostics
        "agents_used": _agents_used(result.get("execution_trace", [])),
        "sub_question_count": len(result.get("sub_questions", []) or []),
        "validation_retries": int(result.get("validation_retries", 0) or 0),
        "retrieval_strategy": result.get("retrieval_strategy", ""),
        "citations_total": len(result.get("citations", []) or []),
        "citations_unverified": len(result.get("unverified_citations", []) or []),
        "confidence_score": result.get("confidence_score"),
        "is_validated": result.get("is_validated"),
    }
    return record


# ── Aggregation ─────────────────────────────────────────────────────────────

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {}

    def _mean(key: str) -> float:
        vals = [r["scores"][key] for r in records]
        return round(sum(vals) / len(vals), 4)

    elapsed = [r.get("elapsed_ms", 0.0) for r in records]
    composites = [r["composite_score"] for r in records]
    passed = sum(1 for r in records if r["passed"])
    verdicts_count: dict = defaultdict(int)
    for r in records:
        verdicts_count[r["judge_verdict"]] += 1

    # Per-source breakdown
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)
    per_source = {}
    for src, rs in by_source.items():
        per_source[src] = {
            "n": len(rs),
            "pass_rate": round(sum(1 for r in rs if r["passed"]) / len(rs), 3),
            "mean_composite": round(sum(r["composite_score"] for r in rs) / len(rs), 4),
            "mean_correctness": round(sum(r["scores"]["correctness"] for r in rs) / len(rs), 4),
        }

    return {
        "n_total": len(records),
        "n_passed": passed,
        "pass_rate": round(passed / len(records), 3),
        "mean_composite": round(sum(composites) / len(composites), 4),
        "mean_correctness": _mean("correctness"),
        "mean_similarity": _mean("similarity"),
        "mean_citation_coverage": _mean("citation_coverage"),
        "mean_faithfulness": _mean("faithfulness"),
        "verdict_distribution": dict(verdicts_count),
        "refusal_rate": round(sum(1 for r in records if r["refused"]) / len(records), 3),
        "mean_sub_questions": round(
            sum(r["sub_question_count"] for r in records) / len(records), 2
        ),
        "mean_validation_retries": round(
            sum(r["validation_retries"] for r in records) / len(records), 2
        ),
        "latency_ms": {
            "mean": round(sum(elapsed) / len(elapsed), 1),
            "p50": round(_percentile(elapsed, 50), 1),
            "p95": round(_percentile(elapsed, 95), 1),
            "max": round(max(elapsed), 1),
        },
        "per_source": per_source,
    }


# ── Main eval loop ──────────────────────────────────────────────────────────

def evaluate_agentic(
    dataset: Optional[Path] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_path: Path = DEFAULT_OUT,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    limit: Optional[int] = None,
    thread_id_prefix: str = "agentic-eval",
) -> dict:
    items = _load_dataset(dataset, data_dir)
    if not items:
        raise ValueError("No QA pairs to evaluate after loading + dedup")

    if limit is not None and limit > 0:
        items = items[:limit]

    logger.info(
        "Agentic eval — %d QA pairs from %s (pass threshold %.2f)",
        len(items),
        dataset.name if dataset else f"{data_dir}/*.json",
        pass_threshold,
    )

    records: list[dict] = []
    pass_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_EVERY = 20

    def _write_results(*, partial: bool, completed: int) -> None:
        """Write the eval JSON. Called every CHECKPOINT_EVERY items and on
        exit (success or failure) so a crash never loses the work already
        done. ``partial=True`` marks the file as incomplete so downstream
        tooling knows whether to trust the summary."""
        snapshot = {
            "config": {
                "dataset": str(dataset) if dataset else f"{data_dir}/*.json",
                "pass_threshold": pass_threshold,
                "weights": WEIGHTS,
            },
            "partial": partial,
            "completed": completed,
            "total": len(items),
            "summary": _aggregate(records) if records else {},
            "records": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)

    pbar = tqdm(
        items,
        desc="Agentic eval",
        unit="q",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    last_completed = 0
    try:
        for i, item in enumerate(pbar, start=1):
            qid = item["qid"]
            pbar.set_postfix_str(f"pass={pass_count}/{i-1} now={qid[:32]}", refresh=False)
            t0 = time.time()
            try:
                result = invoke_query(
                    query=item["question"],
                    thread_id=f"{thread_id_prefix}-{i}",
                    user_id="agentic-eval",
                )
            except Exception as e:
                logger.warning("[%d/%d] %s — workflow raised: %s", i, len(items), qid, e)
                records.append({
                    "qid": qid,
                    "source": item["source"],
                    "question": item["question"],
                    "gold_preview": item["gold_answer"][:200],
                    "predicted_preview": "",
                    "refused": False,
                    "judge_verdict": "error",
                    "judge_reasoning": f"workflow error: {e}",
                    "scores": {k: 0.0 for k in WEIGHTS},
                    "composite_score": 0.0,
                    "passed": False,
                    "agents_used": [],
                    "sub_question_count": 0,
                    "validation_retries": 0,
                    "retrieval_strategy": "",
                    "citations_total": 0,
                    "citations_unverified": 0,
                    "confidence_score": None,
                    "is_validated": False,
                    "elapsed_ms": round((time.time() - t0) * 1000, 1),
                })
                pbar.set_postfix_str(f"pass={pass_count}/{i} last=ERROR", refresh=False)
                last_completed = i
                if i % CHECKPOINT_EVERY == 0:
                    _write_results(partial=True, completed=i)
                continue

            scored = _score_one(item, result, pass_threshold)
            scored["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            records.append(scored)
            if scored["passed"]:
                pass_count += 1

            verdict = "PASS" if scored["passed"] else "FAIL"
            logger.info(
                "[%d/%d] %s [%s] composite=%.2f corr=%.2f sim=%.2f cov=%.2f faith=%.2f t=%.0fms",
                i, len(items), qid, verdict,
                scored["composite_score"],
                scored["scores"]["correctness"],
                scored["scores"]["similarity"],
                scored["scores"]["citation_coverage"],
                scored["scores"]["faithfulness"],
                scored["elapsed_ms"],
            )
            pbar.set_postfix_str(
                f"pass={pass_count}/{i} ({pass_count/i:.0%}) last={verdict} composite={scored['composite_score']:.2f}",
                refresh=False,
            )
            last_completed = i
            if i % CHECKPOINT_EVERY == 0:
                _write_results(partial=True, completed=i)
                logger.info("Checkpoint written → %s (%d/%d records)", out_path, i, len(items))
    except KeyboardInterrupt:
        pbar.close()
        logger.warning("Interrupted — flushing %d completed records to %s", last_completed, out_path)
        _write_results(partial=True, completed=last_completed)
        raise
    except Exception:
        pbar.close()
        logger.exception("Eval crashed — flushing %d completed records to %s", last_completed, out_path)
        _write_results(partial=True, completed=last_completed)
        raise
    pbar.close()

    _write_results(partial=False, completed=len(records))
    summary = _aggregate(records)
    out = {
        "config": {
            "dataset": str(dataset) if dataset else f"{data_dir}/*.json",
            "pass_threshold": pass_threshold,
            "weights": WEIGHTS,
        },
        "partial": False,
        "completed": len(records),
        "total": len(items),
        "summary": summary,
        "records": records,
    }

    _print_report(summary, out_path, pass_threshold, records)
    return out


# ── Console report ──────────────────────────────────────────────────────────

def _print_report(summary: dict, out_path: Path, pass_threshold: float, records: Optional[list[dict]] = None) -> None:
    print()
    print("=" * 80)
    print(f"AGENTIC RAG EVAL — {summary['n_total']} queries (pass threshold: {pass_threshold:.2f})")
    print(f"OVERALL: {summary['n_passed']}/{summary['n_total']} = {summary['pass_rate']:.1%} passed")
    print("=" * 80)

    print("Mean scores")
    print("-" * 80)
    print(f"  composite          : {summary['mean_composite']:.3f}")
    print(f"  correctness (judge): {summary['mean_correctness']:.3f}")
    print(f"  similarity (cos)   : {summary['mean_similarity']:.3f}")
    print(f"  citation_coverage  : {summary['mean_citation_coverage']:.3f}")
    print(f"  faithfulness       : {summary['mean_faithfulness']:.3f}")

    print()
    print("Verdict distribution")
    print("-" * 80)
    for v, n in sorted(summary["verdict_distribution"].items()):
        pct = n / summary["n_total"]
        print(f"  {v:<11}: {n:>3}  ({pct:.1%})")

    print()
    print("Agentic diagnostics")
    print("-" * 80)
    print(f"  refusal_rate           : {summary['refusal_rate']:.1%}")
    print(f"  mean_sub_questions     : {summary['mean_sub_questions']:.2f}")
    print(f"  mean_validation_retries: {summary['mean_validation_retries']:.2f}")

    lat = summary["latency_ms"]
    print()
    print(f"Latency (ms): mean={lat['mean']:.0f}  p50={lat['p50']:.0f}  "
          f"p95={lat['p95']:.0f}  max={lat['max']:.0f}")

    if len(summary.get("per_source", {})) > 1:
        print()
        print("Per source")
        print("-" * 80)
        print(f"  {'source':<40} {'n':>4}  {'pass':>6}  {'compos':>7}  {'corr':>6}")
        for src, m in summary["per_source"].items():
            print(f"  {src:<40} {m['n']:>4}  {m['pass_rate']:>6.1%}  "
                  f"{m['mean_composite']:>7.3f}  {m['mean_correctness']:>6.3f}")

    # Worst-performing questions — sorted by composite score, lowest first.
    # Surfaces specific qids the user can look at in the JSON output.
    if records:
        failures = [r for r in records if not r.get("passed", False)]
        ranked = sorted(failures, key=lambda r: r.get("composite_score", 0.0))[:15]
        if ranked:
            print()
            print("Worst-performing questions (lowest composite first)")
            print("-" * 110)
            print(f"  {'qid':<40} {'compos':>7}  {'corr':>5}  {'sim':>5}  {'cov':>5}  {'faith':>6}  question")
            for r in ranked:
                s = r.get("scores", {})
                q = (r.get("question") or "").replace("\n", " ")
                if len(q) > 60:
                    q = q[:57] + "..."
                print(
                    f"  {r.get('qid','?')[:40]:<40} "
                    f"{r.get('composite_score', 0.0):>7.2f}  "
                    f"{s.get('correctness', 0.0):>5.2f}  "
                    f"{s.get('similarity', 0.0):>5.2f}  "
                    f"{s.get('citation_coverage', 0.0):>5.2f}  "
                    f"{s.get('faithfulness', 0.0):>6.2f}  "
                    f"{q}"
                )

    print()
    print(f"Full per-query results: {out_path}")
    print("=" * 80)
    print()


# ── CLI ─────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="End-to-end agentic RAG evaluation against gold QA pairs."
    )
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Single QA JSON file. If omitted, every *.json under --data-dir is used.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"Directory of QA JSON files (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"Where to write the results JSON (default: {DEFAULT_OUT})")
    parser.add_argument("--pass-threshold", type=float, default=DEFAULT_PASS_THRESHOLD,
                        help=f"Composite score >= this counts as PASS (default: {DEFAULT_PASS_THRESHOLD})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate at most N pairs (handy for smoke tests)")
    args = parser.parse_args()

    setup_logging(level="INFO")
    evaluate_agentic(
        dataset=args.dataset,
        data_dir=args.data_dir,
        out_path=args.out,
        pass_threshold=args.pass_threshold,
        limit=args.limit,
    )


if __name__ == "__main__":
    _cli()
