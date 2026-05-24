"""
Single entry point for every evaluator in this folder.

Subcommands:
  agentic      End-to-end answer quality vs gold QA pairs (QNA_data/*.json).
  retrieval    Retrieval-only: dense vs hybrid vs hybrid+rerank on a gold set.
  adversarial  Robustness / refusal / multi-hop / citation-fidelity edge cases.
  gold         Build the retrieval gold dataset (prerequisite for `retrieval`).
  all          Run agentic + retrieval + adversarial back-to-back.

Examples:
  python -m evaluation.run_all all
  python -m evaluation.run_all agentic --limit 20
  python -m evaluation.run_all agentic --dataset QNA_data/honeywell_esg_qa.json
  python -m evaluation.run_all retrieval --k 1 3 5 10
  python -m evaluation.run_all adversarial
  python -m evaluation.run_all gold --n 50

Each subcommand writes its own results JSON under evaluation/ and prints a
console summary. `all` prints one final combined banner at the end.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ── Lazy delegates ──────────────────────────────────────────────────────────
# We import inside each handler so a missing optional dependency in one
# evaluator doesn't break the whole CLI.

def _run_agentic(args) -> dict:
    from evaluation.agentic_eval import evaluate_agentic, DEFAULT_DATA_DIR, DEFAULT_OUT, DEFAULT_PASS_THRESHOLD
    return evaluate_agentic(
        dataset=args.dataset,
        data_dir=args.data_dir or DEFAULT_DATA_DIR,
        out_path=args.out or DEFAULT_OUT,
        pass_threshold=args.pass_threshold if args.pass_threshold is not None else DEFAULT_PASS_THRESHOLD,
        limit=args.limit,
    )


def _run_retrieval(args) -> dict:
    from evaluation.retrieval_eval import evaluate_retrieval, DEFAULT_GOLD, DEFAULT_OUT, DEFAULT_KS
    return evaluate_retrieval(
        gold_path=args.gold or DEFAULT_GOLD,
        ks=args.k or DEFAULT_KS,
        out_path=args.out or DEFAULT_OUT,
    )


def _run_adversarial(args) -> dict:
    from evaluation.adversarial_eval import evaluate_adversarial, DEFAULT_DATASET, DEFAULT_OUT
    return evaluate_adversarial(
        dataset_path=args.dataset or DEFAULT_DATASET,
        out_path=args.out or DEFAULT_OUT,
    )


def _run_gold(args) -> None:
    from evaluation.build_gold_dataset import build_gold_dataset
    build_gold_dataset(
        n_local=args.n_local,
        n_global=args.n_global,
        group_size=args.group_size,
        pdf_dir=args.pdf_dir,
        out_path=args.out,
    )


def _run_all(args) -> None:
    """Run every evaluator in sequence and print one combined banner."""
    overall_t0 = time.time()
    results: dict[str, dict] = {}

    print("\n" + "#" * 80)
    print("# RUNNING FULL EVALUATION SUITE")
    print("#" * 80)

    stages = [
        ("agentic",     _run_agentic_default),
        ("retrieval",   _run_retrieval_default),
        ("adversarial", _run_adversarial_default),
    ]

    for name, fn in stages:
        print(f"\n>>> [{name}] starting...\n")
        t0 = time.time()
        try:
            results[name] = fn()
            elapsed = time.time() - t0
            print(f"<<< [{name}] finished in {elapsed:.1f}s")
        except FileNotFoundError as e:
            logger.warning("[%s] skipped — %s", name, e)
            results[name] = {"skipped": True, "reason": str(e)}
        except Exception as e:
            logger.exception("[%s] crashed", name)
            results[name] = {"error": str(e)}

    _print_combined(results, time.time() - overall_t0)


def _run_agentic_default() -> dict:
    from evaluation.agentic_eval import evaluate_agentic
    return evaluate_agentic()


def _run_retrieval_default() -> dict:
    from evaluation.retrieval_eval import evaluate_retrieval
    return evaluate_retrieval()


def _run_adversarial_default() -> dict:
    from evaluation.adversarial_eval import evaluate_adversarial
    return evaluate_adversarial()


# ── Combined report (used by `all`) ────────────────────────────────────────

def _print_combined(results: dict, total_elapsed_s: float) -> None:
    print("\n" + "#" * 80)
    print(f"# COMBINED SUMMARY — {total_elapsed_s:.1f}s total")
    print("#" * 80)

    a = results.get("agentic", {})
    if a.get("skipped"):
        print(f"  agentic     : SKIPPED — {a['reason']}")
    elif a.get("error"):
        print(f"  agentic     : ERROR — {a['error']}")
    elif a:
        s = a.get("summary", {})
        print(f"  agentic     : {s.get('n_passed', 0)}/{s.get('n_total', 0)} passed "
              f"({s.get('pass_rate', 0):.1%})  "
              f"composite={s.get('mean_composite', 0):.3f}  "
              f"correctness={s.get('mean_correctness', 0):.3f}")

    r = results.get("retrieval", {})
    if r.get("skipped"):
        print(f"  retrieval   : SKIPPED — {r['reason']}")
    elif r.get("error"):
        print(f"  retrieval   : ERROR — {r['error']}")
    elif r:
        agg = r.get("aggregate_scores", {})
        line = f"  retrieval   : n={r.get('num_queries', 0)}  "
        for stage in ("dense", "hybrid", "hybrid_rerank"):
            if stage in agg:
                line += f"{stage}_mrr={agg[stage].get('mrr', 0):.3f}  "
        print(line.rstrip())

    adv = results.get("adversarial", {})
    if adv.get("skipped"):
        print(f"  adversarial : SKIPPED — {adv['reason']}")
    elif adv.get("error"):
        print(f"  adversarial : ERROR — {adv['error']}")
    elif adv:
        s = adv.get("summary", {})
        print(f"  adversarial : {s.get('overall_passed', 0)}/{s.get('n_total', 0)} passed "
              f"({s.get('overall_pass_rate', 0):.1%})")

    print("#" * 80 + "\n")


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluation.run_all",
        description="Unified entry point for all evaluators in this folder.",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{agentic,retrieval,adversarial,gold,all}")

    # ── agentic ─────────────────────────────────────────────────────────────
    sp = sub.add_parser("agentic", help="End-to-end answer quality vs QNA_data/")
    sp.add_argument("--dataset", type=Path, default=None,
                    help="Single QA JSON file (default: every *.json in --data-dir)")
    sp.add_argument("--data-dir", type=Path, default=None,
                    help="Directory of QA JSON files (default: QNA_data/)")
    sp.add_argument("--out", type=Path, default=None,
                    help="Where to write results (default: evaluation/agentic_eval_results.json)")
    sp.add_argument("--pass-threshold", type=float, default=None,
                    help="Composite score >= this counts as PASS (default: 0.65)")
    sp.add_argument("--limit", type=int, default=None,
                    help="Evaluate at most N pairs")
    sp.set_defaults(func=_run_agentic)

    # ── retrieval ───────────────────────────────────────────────────────────
    sp = sub.add_parser("retrieval", help="Retrieval-only quality (dense / hybrid / +rerank)")
    sp.add_argument("--gold", type=Path, default=None,
                    help="Gold dataset JSONL (default: evaluation/gold_retrieval.jsonl)")
    sp.add_argument("--out", type=Path, default=None,
                    help="Where to write results (default: evaluation/retrieval_eval_results.json)")
    sp.add_argument("--k", type=int, nargs="+", default=None,
                    help="Cutoffs (default: 1 3 5 10)")
    sp.set_defaults(func=_run_retrieval)

    # ── adversarial ─────────────────────────────────────────────────────────
    sp = sub.add_parser("adversarial", help="Robustness / refusal / multi-hop / citation fidelity")
    sp.add_argument("--dataset", type=Path, default=None,
                    help="Adversarial JSONL (default: evaluation/adversarial_questions.jsonl)")
    sp.add_argument("--out", type=Path, default=None,
                    help="Where to write results (default: evaluation/adversarial_eval_results.json)")
    sp.set_defaults(func=_run_adversarial)

    # ── gold ────────────────────────────────────────────────────────────────
    sp = sub.add_parser("gold", help="Build the retrieval gold dataset (one-off, page-by-page)")
    sp.add_argument("--n-local", type=int, default=40,
                    help="Number of local (single-page) questions")
    sp.add_argument("--n-global", type=int, default=10,
                    help="Number of global (multi-page synthesis) questions")
    sp.add_argument("--group-size", type=int, default=4,
                    help="Pages per global-question window")
    sp.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"),
                    help="Folder of PDFs to read")
    sp.add_argument("--out", type=Path, default=Path("evaluation/gold_retrieval.jsonl"),
                    help="Output JSONL path")
    sp.set_defaults(func=_run_gold)

    # ── all ─────────────────────────────────────────────────────────────────
    sp = sub.add_parser("all", help="Run agentic + retrieval + adversarial in sequence")
    sp.set_defaults(func=_run_all)

    return p


def main() -> None:
    setup_logging(level="INFO")
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
