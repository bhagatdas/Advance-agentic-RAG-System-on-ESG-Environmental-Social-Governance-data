"""
Contradiction Detector — scans retrieved chunks for numeric / factual
conflicts about the same metric, year, or claim.

Runs after reasoning, in parallel-spirit with the citation verifier and
faithfulness checker. Surfaces findings as structured ContradictionReport
objects that the API + UI render in a dedicated callout. Unlike the
faithfulness checker (which judges the *answer* against the context), this
agent judges *chunks against each other* — so it can fire even when the
answer is fine.

Cost: one structured LLM call (light model) per query, only when there are
≥2 retrieved chunks AND the answer wasn't a refusal. Skipped otherwise.
"""

from __future__ import annotations

import logging
import time
from utils.tracing import traceable

from graph.state import ContradictionReport
from utils.llm import get_structured_llm

logger = logging.getLogger(__name__)


# Refusal sentinels — when the reasoning agent emits one of these exactly,
# there's nothing meaningful to check for contradictions across.
_REFUSAL_PREFIXES = (
    "I can only answer questions about the ESG reports",
    "This specific information is not available",
)

CONTRADICTION_PROMPT = """You are a strict ESG-report fact-checker. Your job is to find places where the
provided text chunks DISAGREE with each other about the same fact, metric, or
year. Be PRECISE — do not invent contradictions.

CHUNKS (each prefixed with [source: Document, Page N]):
{chunks_text}

SQL TABLE RESULTS (the live structured table data, if any — these are
authoritative for the columns shown):
{sql_text}

YOUR TASK
=========
Find concrete conflicts of the form: "Chunk A says X = a, Chunk B says X = b"
where a ≠ b for the same X (same metric, same year, same scope, same units).

CALIBRATION (apply STRICTLY)
- Differences ≤1% (rounding, formatting) → severity "low" — usually skip unless asked.
- Same value reported in different units (e.g. tonnes vs kilotonnes) → NOT a contradiction; skip.
- Different metrics that share words (e.g. "Scope 1" vs "Scope 1+2") → NOT a contradiction.
- A narrative chunk approximating a precise table number → low severity, only worth flagging if the round-off is >10%.
- A high-severity contradiction is a CLEAR conflict of fact, e.g. "Scope 1 in 2020 = 1,387,727" in one chunk vs "Scope 1 in 2020 = 1,500,000" in another.

If you find NO contradictions, return an empty list. DO NOT pad.

Return your findings:"""


def _format_chunks_for_check(chunks: list[dict], max_chunks: int = 8, max_chars: int = 800) -> str:
    """Build a compact text view of the retrieved chunks for the LLM."""
    if not chunks:
        return "(no chunks)"
    out = []
    for c in chunks[:max_chunks]:
        meta = c.get("metadata", {}) or {}
        doc = meta.get("document_name", "Unknown")
        page = meta.get("page_number", "?")
        content = (c.get("content", "") or "")[:max_chars]
        out.append(f"[source: {doc}, Page {page}]\n{content}")
    return "\n---\n".join(out)


def _format_sql_for_check(sql_results: list[dict] | None, generated_sql: str | None) -> str:
    if not sql_results:
        return "(no SQL results)"
    rows = "\n".join(str(r) for r in sql_results[:20])
    return f"SQL: {generated_sql or '(unknown)'}\nRows:\n{rows}"


@traceable(run_type="chain", name="ContradictionDetector")
def detect_contradictions(
    answer: str,
    retrieved_chunks: list[dict],
    sql_results: list[dict] | None = None,
    generated_sql: str | None = None,
    complexity_tier: str = "moderate",
) -> ContradictionReport:
    """
    Run the contradiction-detection LLM on the retrieved context.

    Skipped (returns empty report) when:
      - complexity_tier == "trivial" (single-fact lookups can't contradict)
      - fewer than 2 retrieved chunks (nothing to compare)
      - the answer was a refusal (no analytical claim was made)
      - all retrieved chunks share the same (document, page)
    """
    if complexity_tier == "trivial":
        logger.info("Contradiction check skipped — complexity_tier=trivial")
        return ContradictionReport(contradictions=[])

    if not retrieved_chunks or len(retrieved_chunks) < 2:
        return ContradictionReport(contradictions=[])

    if answer and any(answer.strip().startswith(p) for p in _REFUSAL_PREFIXES):
        return ContradictionReport(contradictions=[])

    # No contradiction surface when all retrieved chunks come from a single
    # document AND a single page. Single-source narrative can't disagree with
    # itself on a metric, so skip the LLM call entirely.
    sources = {
        (
            (c.get("metadata", {}) or {}).get("document_name", ""),
            (c.get("metadata", {}) or {}).get("page_number", ""),
        )
        for c in retrieved_chunks
    }
    if len(sources) < 2:
        logger.info("Contradiction check skipped — single-source context (%d chunks, 1 page)", len(retrieved_chunks))
        return ContradictionReport(contradictions=[])

    chunks_text = _format_chunks_for_check(retrieved_chunks)
    sql_text = _format_sql_for_check(sql_results, generated_sql)

    try:
        llm = get_structured_llm(ContradictionReport, task_type="light")
        prompt = CONTRADICTION_PROMPT.format(
            chunks_text=chunks_text[:8000],
            sql_text=sql_text[:2000],
        )
        report: ContradictionReport = llm.invoke(prompt)
        # Defensive cap: never surface more than 8 contradictions
        report.contradictions = report.contradictions[:8]
        return report
    except Exception as e:
        logger.warning("Contradiction detection failed: %s", e)
        return ContradictionReport(contradictions=[])


def to_state_dict(report: ContradictionReport) -> dict:
    """Shape the report for AgentState merging + API serialization."""
    return {
        "contradictions": [c.model_dump() for c in report.contradictions],
    }
