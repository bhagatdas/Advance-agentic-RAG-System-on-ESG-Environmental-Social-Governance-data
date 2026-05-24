"""
Adversarial / robustness evaluator.

Targets the failure modes that the recent hardening was designed to catch:

  - wrong_year       : query asks about a year that doesn't appear in the corpus.
                       The metadata-filter + reasoning prompt should produce a
                       refusal (or a clearly hedged answer) — NOT a fabricated
                       value lifted from a different year.

  - missing_data     : query is in-scope but the corpus doesn't contain the
                       answer. The strict reasoning prompt has an exact refusal
                       string for this — we look for that.

  - multi_hop        : compound query that requires retrieving distinct slices
                       of context. The planner is supposed to decompose into
                       ≥2 sub-questions; we score that directly.

  - citation_fidelity: real in-scope queries where the citation verifier should
                       confirm most cited (doc, page) pairs are real. Coverage
                       below the configured threshold is a failure.

Each category gets its own pass/fail criteria — see _score_one().

Run:
    python -m evaluation.adversarial_eval
    python -m evaluation.adversarial_eval --dataset evaluation/adversarial_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from config.settings import settings
from graph.workflow import invoke_query
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("evaluation/adversarial_questions.jsonl")
DEFAULT_OUT = Path("evaluation/adversarial_eval_results.json")

# Substrings (lower-cased) that signal the reasoning agent refused / declined.
# Mirrors the two exact refusal strings in agents/reasoning.py + a couple of
# common hedges so we don't miss a legitimate "I can't answer" answer.
_REFUSAL_MARKERS = [
    "i can only answer questions about the esg reports",
    "outside the scope of the available documents",
    "this specific information is not available",
    "the documents do not contain",
    "not mentioned in the report",
    "not available in the uploaded",
    "is not in the provided context",
]


def _is_refusal(answer: str) -> bool:
    if not answer:
        return True
    low = answer.lower()
    return any(m in low for m in _REFUSAL_MARKERS)


def _score_one(item: dict, result: dict) -> dict:
    """Score one query against its category-specific expectations."""
    category = item.get("category", "")
    expected = item.get("expected", {}) or {}

    answer = result.get("answer", "")
    refused = _is_refusal(answer)
    sub_qs = result.get("sub_questions", []) or []
    coverage = result.get("citation_coverage", 1.0)
    citations = result.get("citations", []) or []
    unverified = result.get("unverified_citations", []) or []

    record = {
        "qid": item.get("qid"),
        "category": category,
        "question": item.get("question"),
        "answer_preview": (answer or "")[:200],
        "refused": refused,
        "sub_question_count": len(sub_qs),
        "citation_coverage": round(coverage, 3),
        "citations_total": len(citations),
        "citations_unverified": len(unverified),
        "passed": False,
        "fail_reason": "",
    }

    if category == "wrong_year":
        # System should have refused rather than fabricated cross-year data.
        if not refused:
            record["fail_reason"] = "system did not refuse for impossible year"
        else:
            record["passed"] = True

    elif category == "missing_data":
        if expected.get("should_refuse", True) and not refused:
            record["fail_reason"] = "expected refusal for off-topic / missing data"
        else:
            record["passed"] = True

    elif category == "multi_hop":
        min_sub = int(expected.get("min_sub_questions", 2))
        if expected.get("should_refuse", False) and not refused:
            # Compound queries shouldn't refuse — flag both directions.
            record["fail_reason"] = "expected refusal but got an answer"
        elif not expected.get("should_refuse", False) and refused:
            record["fail_reason"] = "refused a real compound query"
        elif len(sub_qs) < min_sub:
            record["fail_reason"] = (
                f"planner emitted {len(sub_qs)} sub-questions, expected >= {min_sub}"
            )
        else:
            record["passed"] = True

    elif category == "citation_fidelity":
        min_cov = float(expected.get("min_citation_coverage", settings.citation_coverage_threshold))
        if refused:
            # A refusal on a real in-scope query is suspicious but not a hard
            # citation failure — flag it.
            record["fail_reason"] = "refused a real in-scope query"
        elif coverage < min_cov:
            record["fail_reason"] = (
                f"citation coverage {coverage:.2f} below required {min_cov:.2f}"
            )
        else:
            record["passed"] = True

    else:
        record["fail_reason"] = f"unknown category '{category}'"

    return record


def _aggregate(records: list[dict]) -> dict:
    """Per-category pass rate + a few cross-cutting numbers."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    per_category = {}
    for cat, rs in by_cat.items():
        passed = sum(1 for r in rs if r["passed"])
        per_category[cat] = {
            "n": len(rs),
            "passed": passed,
            "pass_rate": round(passed / len(rs), 3) if rs else 0.0,
            "avg_citation_coverage": round(
                sum(r["citation_coverage"] for r in rs) / len(rs), 3
            ) if rs else 0.0,
            "avg_sub_question_count": round(
                sum(r["sub_question_count"] for r in rs) / len(rs), 2
            ) if rs else 0.0,
            "refusal_rate": round(
                sum(1 for r in rs if r["refused"]) / len(rs), 3
            ) if rs else 0.0,
        }

    overall_passed = sum(1 for r in records if r["passed"])
    return {
        "n_total": len(records),
        "overall_passed": overall_passed,
        "overall_pass_rate": round(overall_passed / len(records), 3) if records else 0.0,
        "per_category": per_category,
    }


def evaluate_adversarial(
    dataset_path: Path = DEFAULT_DATASET,
    out_path: Path = DEFAULT_OUT,
    thread_id_prefix: str = "adv-eval",
) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Adversarial dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if not items:
        raise ValueError(f"Dataset {dataset_path} is empty")

    logger.info("Running adversarial eval — %d queries", len(items))

    records: list[dict] = []
    pass_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_EVERY = 20

    def _write_results(*, partial: bool, completed: int) -> None:
        """Flush eval JSON. Called every CHECKPOINT_EVERY items and on exit
        so a crash never loses the work already done."""
        snapshot = {
            "dataset": str(dataset_path),
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
        desc="Adversarial eval",
        unit="q",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    last_completed = 0
    try:
        for i, item in enumerate(pbar, start=1):
            qid = item.get("qid", f"q_{i}")
            pbar.set_postfix_str(f"pass={pass_count}/{i-1} now={qid[:32]}", refresh=False)
            t0 = time.time()
            try:
                # Fresh thread per query so memory doesn't leak between cases.
                result = invoke_query(
                    query=item["question"],
                    thread_id=f"{thread_id_prefix}-{qid}",
                    user_id="adversarial-eval",
                )
            except Exception as e:
                logger.warning("[%d/%d] %s — workflow raised: %s", i, len(items), qid, e)
                records.append({
                    "qid": qid,
                    "category": item.get("category", ""),
                    "question": item.get("question"),
                    "answer_preview": "",
                    "refused": False,
                    "sub_question_count": 0,
                    "citation_coverage": 0.0,
                    "citations_total": 0,
                    "citations_unverified": 0,
                    "passed": False,
                    "fail_reason": f"workflow error: {e}",
                    "elapsed_ms": (time.time() - t0) * 1000,
                })
                pbar.set_postfix_str(f"pass={pass_count}/{i} last=ERROR", refresh=False)
                last_completed = i
                if i % CHECKPOINT_EVERY == 0:
                    _write_results(partial=True, completed=i)
                continue

            scored = _score_one(item, result)
            scored["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
            records.append(scored)
            if scored["passed"]:
                pass_count += 1

            verdict = "PASS" if scored["passed"] else f"FAIL ({scored['fail_reason']})"
            logger.info(
                "[%d/%d] %s [%s] %s — coverage=%.2f sub_qs=%d refused=%s",
                i, len(items), qid, scored["category"], verdict,
                scored["citation_coverage"], scored["sub_question_count"], scored["refused"],
            )
            pbar.set_postfix_str(
                f"pass={pass_count}/{i} ({pass_count/i:.0%}) last={'PASS' if scored['passed'] else 'FAIL'}",
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
        "dataset": str(dataset_path),
        "partial": False,
        "completed": len(records),
        "total": len(items),
        "summary": summary,
        "records": records,
    }

    _print_report(summary, out_path, records)
    return out


def _print_report(summary: dict, out_path: Path, records: list[dict] | None = None) -> None:
    print()
    print("=" * 78)
    print(f"ADVERSARIAL EVAL — {summary['n_total']} queries")
    print(f"OVERALL PASS RATE: {summary['overall_passed']}/{summary['n_total']} "
          f"= {summary['overall_pass_rate']:.1%}")
    print("=" * 78)
    print(f"{'category':<22} | {'n':>3} | {'pass':>5} | "
          f"{'rate':>6} | {'avg_cov':>7} | {'avg_sub_q':>9} | {'refuse':>6}")
    print("-" * 78)
    for cat, m in summary["per_category"].items():
        print(f"{cat:<22} | {m['n']:>3} | {m['passed']:>5} | "
              f"{m['pass_rate']:>6.1%} | {m['avg_citation_coverage']:>7.2f} | "
              f"{m['avg_sub_question_count']:>9.2f} | {m['refusal_rate']:>6.1%}")
    print("-" * 78)

    # Show every failing question so the user knows what's broken.
    if records:
        failures = [r for r in records if not r.get("passed", False)]
        if failures:
            print()
            print(f"Failing questions ({len(failures)} of {len(records)})")
            print("-" * 110)
            print(f"  {'qid':<28} {'category':<22} {'cov':>5}  reason / question")
            for r in failures:
                q = (r.get("question") or "").replace("\n", " ")
                if len(q) > 50:
                    q = q[:47] + "..."
                reason = r.get("fail_reason") or "—"
                if len(reason) > 24:
                    reason = reason[:21] + "..."
                print(
                    f"  {(r.get('qid','?'))[:28]:<28} "
                    f"{(r.get('category','') or '')[:22]:<22} "
                    f"{r.get('citation_coverage', 0.0):>5.2f}  "
                    f"{reason:<24}  {q}"
                )
            print("-" * 78)

    print(f"Full per-query results written to: {out_path}")
    print()


def _cli():
    parser = argparse.ArgumentParser(description="Adversarial / robustness eval for the RAG pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Adversarial JSONL dataset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Where to write the results JSON")
    args = parser.parse_args()

    setup_logging(level="INFO")
    evaluate_adversarial(dataset_path=args.dataset, out_path=args.out)


if __name__ == "__main__":
    _cli()
