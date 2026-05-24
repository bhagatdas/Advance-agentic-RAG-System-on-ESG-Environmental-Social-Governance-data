"""
Entity-Metric Extractor — pre-synthesis structured fact extraction.

The synthesis-stage failure this addresses: when retrieved chunks contain a
list of entities each with their own numbers (e.g. multiple solar sites with
capacity / generation / offset values), the Reasoning LLM frequently mixes
(entity, value) pairs across rows in its prose. Forcing the LLM to write from
a pre-extracted facts table — instead of free-reading the chunks for numbers —
eliminates row-alignment drift.

Pipeline position: AFTER retrieval (and optionally table_agent), BEFORE
reasoning. The extractor takes the retrieved chunks + any SQL rows and emits
atomic (entity, metric, value, unit, source) facts. The Reasoning prompt
renders these verbatim and is told to source ALL numeric claims from this
table. The deterministic `attribution_verifier` then checks the answer against
the same table after generation.

Two extraction paths in one node:
  1. SQL rows → Facts (DETERMINISTIC, no LLM). Each non-key cell becomes one
     Fact with confidence="high".
  2. Retrieved chunks → Facts (one LIGHT LLM call). Chunks are prefixed with
     unique [CHUNK_<id>] markers and the extractor must cite the marker per
     fact; facts citing unknown chunk IDs are dropped.

Cost: one light LLM call per query, ~2-4s on Ollama. Skipped when
attribution is disabled in settings or complexity_tier=='trivial'.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from utils.tracing import traceable

from graph.state import AgentState, Fact, FactsTable
from utils.llm import (
    get_structured_llm,
    reset_token_counter,
    get_token_counter,
    get_context_budgets,
)
from config.settings import settings
from agents.calculator import _parse_number  # reuse the ESG-aware number parser

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """You are a structured-data extractor for PDF documents. Your job is to pull
atomic (entity, metric, value) facts out of the chunks below — regardless of
the domain (finance, healthcare, legal, scientific, operations, sustainability,
etc.). DO NOT summarize. DO NOT infer. ONE row per (entity, metric) pair.

CHUNKS (each prefixed with [CHUNK_<id>] and [source: Document, Page N]):
{chunks_text}

EXISTING SQL-DERIVED FACTS (already extracted deterministically — do NOT re-emit these):
{sql_facts_summary}

ABSOLUTE RULES (NON-NEGOTIABLE)
===============================
1. ONE Fact = ONE entity ↔ ONE metric ↔ ONE value ↔ ONE unit. NEVER combine
   two entities into one fact. NEVER combine two metrics into one fact.
2. `raw_quote` should be a verbatim span (≤200 chars) from a SINGLE chunk
   that grounds the fact. The IDEAL form has entity + value in one
   adjacent phrase ("Penang, Malaysia generated 1.8 million kWh"). When the
   source is TABLE-SHAPED (see TABLE HANDLING below), the quote may instead
   be the combination of the column-header line and the row-cell line —
   that is a legitimate binding, NOT ambiguous. Set confidence="high" for
   adjacent prose, "medium" for table-derived facts, and "ambiguous" only
   when the binding genuinely cannot be inferred.
3. `source_chunk_id` MUST match one of the [CHUNK_<id>] markers above. Facts
   citing unknown chunk IDs will be discarded.
4. Preserve units VERBATIM. "kWh", "MWh", "GWh", "tCO2e", "Mt", "%" are NOT
   interchangeable. If the source says "1.8 million kWh", value_raw="1.8 million kWh"
   and unit="kWh".
5. Use the entity name as it appears IN THE CHUNK. Do NOT normalize
   ("Chihuahua plant" → "Mexico facility" is WRONG).
6. If two chunks state the SAME fact (same entity + same metric + same value),
   emit it once.
7. If two chunks state DIFFERENT values for the same (entity, metric), emit
   BOTH as separate facts — the contradiction detector will surface the
   conflict later.
8. SKIP facts where the entity binding is unclear (a number floating in prose
   without an unambiguous owner). Better to drop a fact than to guess.
9. Soft cap: at most {max_facts} facts total. If more are present, keep the
   ones most likely to be used in an answer (named entities + concrete numbers).

TABLE HANDLING (CRITICAL — DO NOT MISS TABLE-SHAPED FACTS)
==========================================================
PDFs often present data as ASCII tables where the column headers and the
row values sit on SEPARATE lines, for example:

  <Section title>
  <col_header_1> <col_header_2> <col_header_3> ...
  <Row label A>  <value_A1>     <value_A2>     <value_A3>     ...
  <Row label B>  <value_B1>     <value_B2>     <value_B3>     ...

This is a VALID binding even though the header and the value are on
SEPARATE lines. You MUST emit one fact per (row_label × col_header) cell.

Concrete worked example — a generic time-series table:

  Performance Indicators
                          2018      2019      2020      2021      2022
  Revenue ($M)            142       151       138       156       172
  Headcount               1,205     1,308     1,290     1,330     1,402

You MUST emit 10 facts from this table — for instance:
  - entity="2018", metric="Revenue",   value_raw="142 $M", unit="$M", period="2018"
  - entity="2019", metric="Revenue",   value_raw="151 $M", unit="$M", period="2019"
  - ... (continue through 2022)
  - entity="2018", metric="Headcount", value_raw="1,205",  unit="",   period="2018"
  - ... (continue through 2022)

Guidelines that apply generically across domains:
- When the entity is a YEAR / PERIOD and the metric is a global / corpus-wide
  indicator, use entity=year-as-string and period=same year. The implicit
  "subject" (which company, which document scope) is captured in the metric
  label or the source_doc field — do not invent a subject if the table is
  presenting the report's own primary metric.
- When the entity is a NAMED SUBJECT (a site, a person, a product, a study
  cohort, a contract, a legal case), use that name verbatim as `entity`.
- "X has improved ~70% since <baseline>" / "X grew 15% YoY" → entity="<baseline>",
  metric="<X>", value_raw="~70% improvement", unit="%", period="since <baseline>".
- A clustered set of project / case / cohort results stated together
  ("Program P: 950 actions, $9.5M savings, 42,350 reductions") → emit ONE
  fact per metric, all sharing entity="Program P".
- Per-instance lists ("Site A: …403…, $133,000 savings"; "Site B: …85…")
  → one fact per (instance, metric) pair.

If the chunks visibly contain numeric data and you emit ZERO facts, that is
a FAILURE of this extraction step — re-scan the chunks before returning.

VALUE NORMALIZATION
===================
- value_raw: copy verbatim (e.g. "1,278", "1.8 million kWh", "24%").
- value_numeric: the parsed number with scale words expanded.
    "1.8 million" → 1800000
    "1,278"       → 1278
    "24%"         → 24
    "—" / "N/A"   → null
- unit: the bare unit token ("kWh", "kWp", "%", "tCO2e", "MW"). Empty string
  if the value is unitless (e.g. counts, ratios reported as "0.45").

PERIOD / TIME
=============
- Pull the period from nearby text when available: "in 2023", "FY2022 baseline",
  "annually", "year-to-date 2024". Use null when no period is stated.

ENTITY_TYPE
===========
Use one of: site, facility, country, region, business_unit, scope (e.g. Scope 1),
metric_group, product, project, other.

If the chunks contain NO numerical facts (pure narrative / qualitative), return
an empty facts list. Do NOT pad.

Now extract:"""


def _is_refusal_or_empty(answer: str) -> bool:
    """Mirror the contradiction-detector / validation refusal check."""
    if not answer:
        return False  # extractor runs BEFORE reasoning — answer is empty
    low = answer.lower()
    return (
        "i can only answer questions about the esg reports" in low
        or "this specific information is not available" in low
    )


def _format_chunks_with_ids(chunks: list[dict], char_budget: int) -> tuple[str, dict[str, dict]]:
    """Render chunks with [CHUNK_<id>] markers so the extractor can cite them.

    Returns (rendered_text, id_to_chunk_map). Chunks are deduped (parent
    supersedes child) and ordered by document/page, mirroring the renderer
    used in reasoning.py — this keeps the extractor and the reasoning model
    looking at the same text.
    """
    if not chunks:
        return "(no chunks)", {}

    # Dedup: drop children whose parent is in the set
    parent_ids_present: set[str] = set()
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        if meta.get("chunk_type") == "parent" or c.get("is_parent_context"):
            pid = meta.get("parent_id") or c.get("chunk_id") or ""
            if pid:
                parent_ids_present.add(pid)

    deduped: list[dict] = []
    seen: set[str] = set()
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        cid = c.get("chunk_id") or meta.get("chunk_id") or ""
        if cid and cid in seen:
            continue
        if meta.get("chunk_type") == "child":
            pid = meta.get("parent_id", "")
            if pid and pid in parent_ids_present:
                continue
        deduped.append(c)
        if cid:
            seen.add(cid)

    def _key(c: dict) -> tuple:
        meta = c.get("metadata", {}) or {}
        doc = (meta.get("document_name") or "").lower()
        try:
            page = int(meta.get("page_number") or 0)
        except (TypeError, ValueError):
            page = 0
        return (doc, page)

    ordered = sorted(deduped, key=_key)

    out: list[str] = []
    id_map: dict[str, dict] = {}
    used = 0
    for idx, c in enumerate(ordered):
        # Short, monotonic, prompt-friendly id
        short_id = f"C{idx + 1}"
        meta = c.get("metadata", {}) or {}
        doc = meta.get("document_name", "Unknown")
        page = meta.get("page_number", "?")
        content = (c.get("content", "") or "").strip()
        # Cap per chunk so a single mega-chunk can't blow the budget
        snippet = content[:1500]
        block = f"[CHUNK_{short_id}] [source: {doc}, Page {page}]\n{snippet}"
        if used + len(block) > char_budget:
            break
        out.append(block)
        id_map[short_id] = c
        used += len(block) + 2  # +2 for the join separator

    return "\n\n".join(out) if out else "(no chunks fit in budget)", id_map


def _facts_from_sql(
    sql_results: list[dict],
    generated_sql: str,
    max_rows: int = 30,
) -> list[Fact]:
    """Deterministic: every SQL row → N Facts (one per non-key cell).

    The first column is treated as the entity identifier (typical pattern for
    the Table Agent's queries, which already prefer `entity_col, *`). Each
    subsequent column becomes a metric for that entity.
    """
    if not sql_results:
        return []

    facts: list[Fact] = []
    table_hint = _table_name_from_sql(generated_sql) or "table"

    for row_idx, row in enumerate(sql_results[:max_rows]):
        if not isinstance(row, dict) or not row:
            continue
        cols = list(row.keys())
        if not cols:
            continue
        entity_col = cols[0]
        entity_val = str(row[entity_col]).strip()
        if not entity_val:
            continue

        for col in cols[1:]:
            cell = row[col]
            if cell is None:
                continue
            value_raw = str(cell).strip()
            if not value_raw or value_raw in {"—", "-", "N/A", "n/a"}:
                continue

            # Parse value + try to peel off a unit suffix
            value_numeric = _parse_number(value_raw)
            unit = _extract_unit_from_column_or_value(col, value_raw)
            period = _period_from_column_name(col)

            metric_label = _humanize_column_name(col)
            facts.append(Fact(
                entity=entity_val,
                entity_type=_guess_entity_type(entity_col, entity_val),
                metric=metric_label,
                value_raw=value_raw,
                value_numeric=value_numeric,
                unit=unit,
                period=period,
                source_chunk_id=f"sql:{table_hint}:{row_idx}",
                source_doc=f"SQL[{table_hint}]",
                source_page=0,
                raw_quote=f"{entity_col}={entity_val}; {col}={value_raw}",
                confidence="high",
            ))
    return facts


# ── small helpers for SQL row → Fact promotion ────────────────────────────

_TABLE_FROM_SQL = re.compile(r"\bfrom\s+\"?([a-zA-Z0-9_]+)\"?", re.IGNORECASE)
_YEAR_IN_NAME = re.compile(r"(19|20)\d{2}")
_UNIT_SUFFIX = re.compile(r"([a-zA-Zµ%/]+)$")


def _table_name_from_sql(sql: str) -> Optional[str]:
    if not sql:
        return None
    m = _TABLE_FROM_SQL.search(sql)
    return m.group(1) if m else None


def _humanize_column_name(col: str) -> str:
    """`col_2018` / `fy_2018` / `installed_capacity_kwp` → human-readable."""
    name = col.replace("_", " ").strip()
    return name


def _period_from_column_name(col: str) -> Optional[str]:
    m = _YEAR_IN_NAME.search(col or "")
    return m.group(0) if m else None


def _extract_unit_from_column_or_value(col: str, value_raw: str) -> str:
    """Pull a unit token off the column name or value suffix."""
    # Common ESG column suffix patterns
    low = (col or "").lower()
    for u in ("kwh", "mwh", "gwh", "kwp", "mw", "tco2e", "tco2", "mtco2e",
              "tonnes", "tons", "kg", "m3", "%", "pct"):
        if low.endswith("_" + u) or low.endswith(u):
            if u in ("pct",):
                return "%"
            return u.upper() if u in ("kwh", "mwh", "gwh", "kwp", "mw") else u
    # Else try the value suffix
    if value_raw.endswith("%"):
        return "%"
    m = _UNIT_SUFFIX.search(value_raw.replace(",", ""))
    if m:
        token = m.group(1)
        if token.isalpha() and 1 <= len(token) <= 6:
            return token
    return ""


def _guess_entity_type(entity_col: str, entity_val: str) -> str:
    col = (entity_col or "").lower()
    if any(k in col for k in ("site", "plant", "facility", "location")):
        return "site"
    if "country" in col or "region" in col:
        return "country"
    if "scope" in col or "scope" in entity_val.lower():
        return "scope"
    if "unit" in col or "business" in col:
        return "business_unit"
    return "other"


# ── extractor node ─────────────────────────────────────────────────────────


@traceable(run_type="chain", name="EntityMetricExtractor")
def entity_metric_extractor_node(state: AgentState) -> dict:
    """Extract structured (entity, metric, value) facts BEFORE reasoning runs."""
    start = time.time()
    reset_token_counter()

    if not settings.attribution_enabled:
        return _skip(start, reason="attribution_disabled")

    # Skip on trivial single-fact lookups — the cost outweighs the benefit
    if state.get("complexity_tier") == "trivial":
        return _skip(start, reason="complexity_tier=trivial")

    retrieved_chunks = state.get("retrieved_chunks", []) or []
    sql_results = state.get("sql_results", []) or []
    generated_sql = state.get("generated_sql", "") or ""

    # Nothing to extract from
    if not retrieved_chunks and not sql_results:
        return _skip(start, reason="no_context")

    # 1. SQL → Facts (deterministic)
    sql_facts = _facts_from_sql(sql_results, generated_sql)

    # 2. Chunks → Facts (one LLM call)
    chunk_facts: list[Fact] = []
    id_map: dict[str, dict] = {}
    if retrieved_chunks:
        budgets = get_context_budgets("light")
        # Use ~75% of the text budget for chunks; leave room for the prompt
        # template, SQL facts summary, and the structured-output schema overhead.
        chunk_budget = int(budgets["text_context"] * 0.75)
        chunks_text, id_map = _format_chunks_with_ids(retrieved_chunks, chunk_budget)
        sql_facts_summary = _summarize_sql_facts(sql_facts)
        try:
            llm = get_structured_llm(FactsTable, task_type="light")
            prompt = EXTRACTION_PROMPT.format(
                chunks_text=chunks_text,
                sql_facts_summary=sql_facts_summary,
                max_facts=settings.attribution_max_facts,
            )
            table: FactsTable = llm.invoke(prompt)
            chunk_facts = list(table.facts or [])
            ambiguous_count = int(table.ambiguous_count or 0)
        except Exception as e:
            logger.warning("Entity-metric extraction failed: %s", e)
            chunk_facts = []
            ambiguous_count = 0
    else:
        ambiguous_count = 0

    # 3. Validate, filter, dedupe
    valid_chunk_facts = _validate_chunk_facts(chunk_facts, id_map)
    all_facts = sql_facts + valid_chunk_facts
    all_facts = _dedupe_facts(all_facts)
    all_facts = all_facts[: settings.attribution_max_facts]

    elapsed = (time.time() - start) * 1000
    facts_dump = [f.model_dump() for f in all_facts]
    high_conf = sum(1 for f in all_facts if f.confidence == "high")

    logger.info(
        "Entity-metric extraction — facts=%d (high=%d, sql=%d, chunk=%d), ambiguous=%d, %.0fms",
        len(all_facts), high_conf, len(sql_facts), len(valid_chunk_facts),
        ambiguous_count, elapsed,
    )

    return {
        "facts": facts_dump,
        "facts_ambiguous_count": ambiguous_count,
        "execution_trace": [{
            "agent": "EntityMetricExtractor",
            "action": "extract_facts",
            "input_summary": (
                f"chunks={len(retrieved_chunks)}, sql_rows={len(sql_results)}"
            ),
            "output_summary": (
                f"facts={len(all_facts)} (sql={len(sql_facts)}, chunk={len(valid_chunk_facts)}), "
                f"ambiguous={ambiguous_count}"
            ),
            "duration_ms": round(elapsed, 1),
            "tokens": get_token_counter(),
        }],
    }


def _skip(start: float, reason: str) -> dict:
    elapsed = (time.time() - start) * 1000
    logger.info("Entity-metric extraction skipped — %s", reason)
    return {
        "facts": [],
        "facts_ambiguous_count": 0,
        "execution_trace": [{
            "agent": "EntityMetricExtractor",
            "action": "skip",
            "input_summary": reason,
            "output_summary": "skipped",
            "duration_ms": round(elapsed, 1),
            "tokens": get_token_counter(),
        }],
    }


def _summarize_sql_facts(sql_facts: list[Fact]) -> str:
    """Compact view of SQL-derived facts so the LLM doesn't re-emit them."""
    if not sql_facts:
        return "(none)"
    lines = []
    for f in sql_facts[:30]:
        lines.append(f"- {f.entity} | {f.metric} = {f.value_raw} ({f.unit or 'unitless'})")
    if len(sql_facts) > 30:
        lines.append(f"...and {len(sql_facts) - 30} more SQL rows")
    return "\n".join(lines)


def _validate_chunk_facts(facts: list[Fact], id_map: dict[str, dict]) -> list[Fact]:
    """Drop facts that cite unknown chunk IDs OR have ambiguous binding.

    Replaces the prompt-local marker (e.g. "C3") with the chunk's REAL
    chunk_id so downstream consumers (per-sub-question evidence routing,
    attribution_verifier) can match facts back to their source chunk.
    """
    out: list[Fact] = []
    for f in facts:
        if f.confidence == "ambiguous":
            continue  # ambiguous facts only inflate the counter; don't render
        # Strip a "CHUNK_" prefix if the LLM included it
        cid = (f.source_chunk_id or "").strip()
        if cid.startswith("CHUNK_"):
            cid = cid[len("CHUNK_"):]
        chunk = id_map.get(cid)
        if chunk is None:
            # Try last-resort: maybe the LLM emitted "C3" already as expected
            chunk = id_map.get(f.source_chunk_id or "")
        if chunk is None:
            logger.debug("Dropping fact with unknown chunk_id=%r entity=%r",
                         f.source_chunk_id, f.entity)
            continue
        meta = chunk.get("metadata", {}) or {}
        # Patch doc/page from the actual chunk metadata so they're authoritative,
        # regardless of what the LLM wrote. Replace the prompt-local "C<N>"
        # marker with the chunk's REAL chunk_id so per-sub-question routing
        # can map facts back to their source chunk in O(1).
        real_cid = chunk.get("chunk_id") or meta.get("chunk_id") or cid
        f.source_chunk_id = real_cid
        f.source_doc = meta.get("document_name") or f.source_doc or "Unknown"
        try:
            f.source_page = int(meta.get("page_number") or f.source_page or 0)
        except (TypeError, ValueError):
            pass
        # Re-parse value_numeric defensively (the LLM sometimes drops the scale)
        if f.value_numeric is None and f.value_raw:
            f.value_numeric = _parse_number(f.value_raw)
        out.append(f)
    return out


def _dedupe_facts(facts: list[Fact]) -> list[Fact]:
    """Drop exact (entity, metric, value, unit, period) duplicates."""
    seen: set[tuple] = set()
    out: list[Fact] = []
    for f in facts:
        key = (
            (f.entity or "").strip().lower(),
            (f.metric or "").strip().lower(),
            (f.value_raw or "").strip().lower(),
            (f.unit or "").strip().lower(),
            (f.period or "") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# ── facts table rendering (used by reasoning.py) ───────────────────────────


def facts_for_sub_question(
    facts: list[dict],
    sub_question_chunks: list[dict],
) -> list[dict]:
    """Subset of facts whose source chunk falls in `sub_question_chunks`.

    Matches by chunk_id AND parent_id, since the extractor's facts cite the
    leaf chunk_id while the per-sub-question scope often pulls in the parent
    too. Returns facts in the same order as the global facts list (preserves
    confidence ordering).
    """
    if not facts or not sub_question_chunks:
        return []
    in_scope: set[str] = set()
    for c in sub_question_chunks:
        meta = c.get("metadata", {}) or {}
        for key in (c.get("chunk_id"), meta.get("chunk_id"), meta.get("parent_id")):
            if key:
                in_scope.add(key)
    return [f for f in facts if (f.get("source_chunk_id") or "") in in_scope]


def render_facts_for_prompt(facts: list[dict]) -> str:
    """Render the facts table into explicit row-delimited blocks for the
    Reasoning prompt. NOT markdown — LLMs lose alignment on markdown tables
    with many columns. One `=== FACT N ===` block per fact, with a citation
    line the model can copy verbatim.
    """
    if not facts:
        return "(no facts extracted — answer from prose only)"
    out: list[str] = []
    for idx, f in enumerate(facts, start=1):
        doc = f.get("source_doc") or "Unknown"
        page = f.get("source_page") or 0
        citation = f"[{doc}, Page {page}]"
        out.append(
            "\n".join([
                f"=== FACT {idx} ===  citation: {citation}",
                f"entity:        {f.get('entity', '')}",
                f"metric:        {f.get('metric', '')}",
                f"value_raw:     {f.get('value_raw', '')}",
                f"value_numeric: {f.get('value_numeric')}",
                f"unit:          {f.get('unit', '')}",
                f"period:        {f.get('period') or '-'}",
                f"raw_quote:     \"{(f.get('raw_quote') or '')[:200]}\"",
            ])
        )
    return "\n\n".join(out)
