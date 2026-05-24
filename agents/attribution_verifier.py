"""
Attribution Verifier — deterministic post-pass that catches entity ↔ value
mis-attribution in the Reasoning agent's answer.

Background
==========
The Reasoning LLM often pairs the wrong number with the wrong entity when
synthesizing prose from chunks containing list-like data. Example failure
(real, from production):

    PDF says:
      - Penang, Malaysia  → 1278 kWp → 1.8 million kWh annually → 24% offset
      - Chonburi, Thailand → 703 kWp → 820,000 kWh annually
      - Chihuahua, Mexico  → 2725 kWp → 4.8 million kWh annually

    Answer wrongly stated:
      - Mexico generates 1.8 million kWh, offsetting 24% of site energy.
      (those values belonged to Penang.)

The existing verifiers don't catch this:
  - citation_verifier checks (doc, page) only.
  - faithfulness_checker judges claims against context as a whole — both
    "Mexico" and "1.8M kWh" appear in the context, just in different rows.
  - contradiction_detector looks at chunks vs. chunks, not answer vs. chunks.

This module
===========
Runs AFTER reasoning, no LLM call. For every numeric mention in the answer:
  1. Find the bound entity by token-proximity scan around the number.
  2. Look up the (entity, value, unit) triple in the facts table produced by
     the entity_metric_extractor.
  3. Classify the failure mode:
       wrong_entity      — value matches a fact bound to a DIFFERENT entity
       no_supporting_fact — no fact within tolerance for entity OR value
       unit_mismatch     — value+entity match a fact but unit differs
       value_mismatch    — entity matches a fact but value is off
  4. Emit AttributionReport with score = verified / total.

Reused primitives: `agents.calculator._parse_number` for ESG-aware number
parsing (handles "1.8 million", "$9.5M", "8.14%").

Cost: ~10-50ms on a typical answer. Pure Python. No new dependencies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from graph.state import AttributionMismatch, AttributionReport, Fact
from agents.calculator import _parse_number, _SCALE
from config.settings import settings

logger = logging.getLogger(__name__)


# ── number + unit token extraction ─────────────────────────────────────────

# Number with optional commas + decimals, plus an optional scale word and unit.
# Captures the full literal so we can re-parse with calculator._parse_number.
#
# Scale-word note: single-letter scales (k / m / b / t) are ONLY treated as
# scale words when not followed by another letter — otherwise "1278 kWp" would
# be parsed as "1278 k" = 1,278,000 (with "Wp" left dangling). The negative
# lookahead `(?![a-zA-Z])` blocks the bad case.
_NUMBER_WITH_UNIT_RE = re.compile(
    r"""
    (?P<full>
      (?P<sign>[+-]?)
      (?P<digits>\d{1,3}(?:,\d{3})+|\d+)
      (?P<dec>\.\d+)?
      \s*
      (?P<scale>(?:million|billion|thousand|trillion|mn|bn|(?:[kmbt])(?![a-zA-Z])))?
      \s*
      (?P<unit>kWh|MWh|GWh|kWp|MW|GW|tCO2e|tCO2|MtCO2e|tonnes|tons|kg|m3|kg/CO2e|%|percent)?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Inline-citation marker `[Doc, Page N]` — numbers INSIDE these aren't claims.
_INLINE_CITATION_RE = re.compile(
    r"\[[^\[\]]+?,\s*(?:page|p\.?)\s*\d+\]",
    flags=re.IGNORECASE,
)

# Markdown delta-table rows — calculator already verified those, don't double-count.
# We detect them by looking at the line containing the match.
_DELTA_TABLE_DIVIDER_RE = re.compile(r"^\|?[\s:\-|]+\|?$")
_DELTA_HEADER_CUES = ("δ absolute", "δ relative", "delta", "change")


# ── unit canonicalization ──────────────────────────────────────────────────

# Tokens that mean the same thing for matching purposes. We do NOT collapse
# kWh/MWh/GWh — different magnitudes, different facts.
_UNIT_ALIASES = {
    "kwh": "kwh", "kilowatt-hours": "kwh", "kilowatt-hour": "kwh", "kwhrs": "kwh",
    "mwh": "mwh", "megawatt-hours": "mwh",
    "gwh": "gwh", "gigawatt-hours": "gwh",
    "kwp": "kwp", "kilowatt-peak": "kwp",
    "mw": "mw", "gw": "gw",
    "tco2e": "tco2e", "tco2": "tco2", "mtco2e": "mtco2e",
    "tonnes": "tonnes", "tons": "tonnes", "t": "tonnes",
    "%": "%", "percent": "%", "pct": "%",
    "": "",
}


def _canon_unit(u: str) -> str:
    if not u:
        return ""
    return _UNIT_ALIASES.get(u.strip().lower(), u.strip().lower())


# ── entity proximity scan ──────────────────────────────────────────────────


def _build_entity_index(facts: list[dict]) -> list[tuple[str, str]]:
    """Return [(canonical_lower, original)] of all unique entity strings in the
    facts table — used for proximity matching in the answer text."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for f in facts:
        ent = (f.get("entity") or "").strip()
        if not ent:
            continue
        key = ent.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((key, ent))
    return out


def _sentence_bounds(answer: str, pos: int) -> tuple[int, int]:
    """Return [start, end] char offsets for the sentence containing `pos`.

    A sentence ends at `.`, `!`, `?` (followed by whitespace) OR at a newline.
    Used as the primary scope for entity ↔ value binding — entities and values
    that share a sentence are far more likely to be related than ones that
    share only a paragraph.
    """
    if not answer:
        return 0, 0
    n = len(answer)
    # Look backward for a sentence boundary
    start = 0
    for i in range(min(pos, n - 1), -1, -1):
        ch = answer[i]
        if ch == "\n":
            start = i + 1
            break
        if ch in ".!?" and i + 1 < n and answer[i + 1].isspace():
            start = i + 1
            # skip the whitespace
            while start < n and answer[start].isspace():
                start += 1
            break
    # Look forward
    end = n
    for i in range(pos, n):
        ch = answer[i]
        if ch == "\n":
            end = i
            break
        if ch in ".!?" and (i + 1 >= n or answer[i + 1].isspace()):
            end = i + 1
            break
    return start, end


def _find_entity_near(
    answer: str,
    span_start: int,
    span_end: int,
    entity_index: list[tuple[str, str]],
    proximity_chars: int = 240,
) -> Optional[str]:
    """Find the entity bound to the number at [span_start, span_end].

    Two-tier scope:
      1. SENTENCE scope (primary). The enclosing sentence is the natural
         binding unit. If exactly one entity appears, return it. If multiple,
         return the closest to the number.
      2. WINDOW scope (fallback). If no entity is in the sentence, look in a
         ±proximity_chars window around the number — same closest-wins logic.

    Returns the ORIGINAL-case entity string, or None.
    """
    if not entity_index:
        return None
    if not answer:
        return None

    def _scan(text_start: int, text_end: int) -> Optional[str]:
        window = answer[text_start:text_end].lower()
        if not window:
            return None
        num_pos_in_window = span_start - text_start
        best: tuple[float, str] | None = None
        for key, original in entity_index:
            idx = 0
            while True:
                hit = window.find(key, idx)
                if hit == -1:
                    break
                entity_center = hit + len(key) // 2
                distance = abs(entity_center - num_pos_in_window)
                # Tiebreaker: prefer entity BEFORE the number (lower hit pos),
                # which is the typical ESG-prose ordering.
                score = distance + (5 if hit > num_pos_in_window else 0)
                if best is None or score < best[0]:
                    best = (score, original)
                idx = hit + 1
        return best[1] if best else None

    # 1. Sentence scope
    s_start, s_end = _sentence_bounds(answer, span_start)
    found = _scan(s_start, s_end)
    if found:
        return found

    # 2. Window scope (fallback)
    w_start = max(0, span_start - proximity_chars)
    w_end = min(len(answer), span_end + proximity_chars)
    return _scan(w_start, w_end)


def _entity_fuzzy_match(a: str, b: str, ratio_threshold: float) -> bool:
    """Two entities match if either contains the other (case-insensitive)
    OR SequenceMatcher.ratio() ≥ threshold. The substring check catches
    'Penang' vs 'Penang, Malaysia'; the ratio check catches minor variants."""
    if not a or not b:
        return False
    la, lb = a.lower().strip(), b.lower().strip()
    if la == lb:
        return True
    if la in lb or lb in la:
        return True
    return SequenceMatcher(None, la, lb).ratio() >= ratio_threshold


def _values_close(a: Optional[float], b: Optional[float], rel_tol: float) -> bool:
    if a is None or b is None:
        return False
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) <= 1.0
    return abs(a - b) / max(abs(a), abs(b)) <= rel_tol


# ── delta-table line detection (skip claims calculator already covers) ────


def _build_skip_line_set(answer: str) -> set[int]:
    """Return the set of line indices that are inside a 5-column delta table.

    Calculator already verified those rows; numbers there must not be
    double-counted as attribution risks. We don't need to be surgical — we
    just need to know "this line is a delta-table row".
    """
    lines = answer.split("\n")
    skip: set[int] = set()
    i = 0
    while i < len(lines) - 1:
        header = lines[i]
        if "|" not in header:
            i += 1
            continue
        divider = lines[i + 1].strip()
        if not _DELTA_TABLE_DIVIDER_RE.match(divider) or "-" not in divider:
            i += 1
            continue
        header_cells = [c.strip() for c in header.strip().strip("|").split("|")]
        if len(header_cells) != 5:
            i += 1
            continue
        last_two = " ".join(header_cells[-2:]).lower()
        if not any(cue in last_two for cue in _DELTA_HEADER_CUES) and not (
            "absolute" in last_two and "relative" in last_two
        ):
            i += 1
            continue
        # Mark header + divider + body rows
        j = i + 2
        while j < len(lines):
            row = lines[j]
            if "|" not in row or row.strip() == "":
                break
            skip.add(j)
            j += 1
        skip.add(i)
        skip.add(i + 1)
        i = j
    return skip


# ── line-offset cache (so we can map char-span → line index) ──────────────


def _char_offsets_to_line(answer: str) -> list[int]:
    """Cumulative char offset for each line in the answer. Used to map a
    regex match position to its line index (for the delta-skip set)."""
    offsets: list[int] = []
    pos = 0
    for line in answer.split("\n"):
        offsets.append(pos)
        pos += len(line) + 1  # +1 for the newline
    return offsets


def _line_of_pos(line_offsets: list[int], pos: int) -> int:
    # Binary search since offsets are monotonic
    lo, hi = 0, len(line_offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_offsets[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ── citation-marker masking ────────────────────────────────────────────────


def _mask_citations(answer: str) -> str:
    """Replace `[Doc, Page N]` markers with spaces so the page number isn't
    treated as a numeric claim. Keeps character offsets stable."""
    def _repl(m: re.Match) -> str:
        return " " * (m.end() - m.start())
    return _INLINE_CITATION_RE.sub(_repl, answer)


# ── core: find a sentence around a position ───────────────────────────────


_SENTENCE_END = re.compile(r"[.!?]\s")


def _enclosing_sentence(answer: str, pos: int, max_chars: int = 240) -> str:
    """Best-effort sentence around a position. Caps length and falls back to
    surrounding window if no sentence boundary is found."""
    start = max(0, pos - max_chars)
    end = min(len(answer), pos + max_chars)
    window = answer[start:end]
    # Find last sentence end before pos
    rel_pos = pos - start
    pre = window[:rel_pos]
    post = window[rel_pos:]
    pre_end = max(pre.rfind(". "), pre.rfind("! "), pre.rfind("? "), pre.rfind("\n"))
    if pre_end >= 0:
        sent_start = start + pre_end + 1
    else:
        sent_start = start
    post_end = post.find(". ")
    if post_end < 0:
        post_end = post.find("\n")
    if post_end >= 0:
        sent_end = pos + post_end + 1
    else:
        sent_end = end
    return answer[sent_start:sent_end].strip()


# ── main entry point ──────────────────────────────────────────────────────


@dataclass
class _NumericClaim:
    full: str
    value_numeric: Optional[float]
    unit: str
    span_start: int
    span_end: int


def _extract_numeric_claims(
    answer: str,
    skip_lines: set[int],
    line_offsets: list[int],
) -> list[_NumericClaim]:
    """Find every numeric token in `answer` outside citations + delta-table
    rows. The caller is responsible for masking citations BEFORE calling."""
    claims: list[_NumericClaim] = []
    for m in _NUMBER_WITH_UNIT_RE.finditer(answer):
        full = m.group("full")
        if not full or not any(ch.isdigit() for ch in full):
            continue
        # Skip rows inside delta tables
        line_idx = _line_of_pos(line_offsets, m.start())
        if line_idx in skip_lines:
            continue
        # Skip trivial standalone integers ≤9 (almost never ESG metrics; usually
        # list numbering, "3 sites", year counts, etc.) UNLESS they carry a unit
        # or scale word.
        if (
            not m.group("scale")
            and not m.group("unit")
            and not m.group("dec")
            and "," not in (m.group("digits") or "")
        ):
            try:
                if abs(int(m.group("digits"))) <= 9:
                    continue
            except (TypeError, ValueError):
                pass

        value_numeric = _parse_number(full)
        unit = (m.group("unit") or "").strip()
        if not unit and "%" in full:
            unit = "%"
        claims.append(_NumericClaim(
            full=full.strip(),
            value_numeric=value_numeric,
            unit=unit,
            span_start=m.start(),
            span_end=m.end(),
        ))
    return claims


def verify_attribution(
    answer: str,
    facts: list[dict],
) -> AttributionReport:
    """Verify that every numeric claim in the answer resolves against the
    facts table to the CORRECT entity.

    Returns an AttributionReport with mismatches + score in [0, 1].
    """
    if not answer or not answer.strip():
        return AttributionReport()
    if not facts:
        # No facts to verify against — can't check attribution. Treat as
        # neutral pass (1.0) so we don't block answers when extraction
        # legitimately produced no facts (e.g. pure narrative queries).
        return AttributionReport()

    masked = _mask_citations(answer)
    line_offsets = _char_offsets_to_line(masked)
    skip_lines = _build_skip_line_set(masked)
    claims = _extract_numeric_claims(masked, skip_lines, line_offsets)
    if not claims:
        return AttributionReport()

    entity_index = _build_entity_index(facts)
    fuzzy_ratio = settings.attribution_entity_fuzzy_ratio
    val_tol = settings.attribution_value_tolerance
    prox_chars = max(20, settings.attribution_proximity_tokens * 6)  # ~6 chars/token

    mismatches: list[AttributionMismatch] = []
    verified = 0

    for c in claims:
        # 1. Find which entity is bound to this number in the answer
        claimed_entity = _find_entity_near(
            answer=masked,
            span_start=c.span_start,
            span_end=c.span_end,
            entity_index=entity_index,
            proximity_chars=prox_chars,
        )

        # 2. Find facts with a matching value (within tolerance)
        value_matches: list[dict] = []
        for f in facts:
            if _values_close(c.value_numeric, f.get("value_numeric"), val_tol):
                value_matches.append(f)

        if not value_matches:
            # No fact even has this value — claim is unsupported.
            # If the user intentionally cited a value from the source, this
            # may still be fine (rounding, formatting) — but we flag it.
            mismatches.append(AttributionMismatch(
                claim_sentence=_enclosing_sentence(answer, c.span_start),
                claimed_entity=claimed_entity or "(unknown)",
                claimed_value_raw=c.full,
                claimed_value_numeric=c.value_numeric,
                claimed_unit=c.unit,
                nearest_fact=None,
                failure_mode="no_supporting_fact",
            ))
            continue

        # 3. Among value-matching facts, do any have the right entity?
        if claimed_entity:
            entity_aligned = [
                f for f in value_matches
                if _entity_fuzzy_match(claimed_entity, f.get("entity", ""), fuzzy_ratio)
            ]
        else:
            entity_aligned = []

        if entity_aligned:
            # 4. Check the unit on the aligned fact(s)
            unit_ok = any(
                _canon_unit(c.unit) == _canon_unit(f.get("unit", ""))
                or not c.unit  # answer didn't write a unit — accept the fact's
                or not f.get("unit")  # fact has no unit — likely unitless number
                for f in entity_aligned
            )
            if unit_ok:
                verified += 1
                continue
            # Entity + value match but unit doesn't — flag
            best = entity_aligned[0]
            mismatches.append(AttributionMismatch(
                claim_sentence=_enclosing_sentence(answer, c.span_start),
                claimed_entity=claimed_entity,
                claimed_value_raw=c.full,
                claimed_value_numeric=c.value_numeric,
                claimed_unit=c.unit,
                nearest_fact=Fact(**best),
                failure_mode="unit_mismatch",
            ))
            continue

        # 5. Value matches a fact, but for a DIFFERENT entity → THE BUG.
        best = value_matches[0]
        mismatches.append(AttributionMismatch(
            claim_sentence=_enclosing_sentence(answer, c.span_start),
            claimed_entity=claimed_entity or "(unknown)",
            claimed_value_raw=c.full,
            claimed_value_numeric=c.value_numeric,
            claimed_unit=c.unit,
            nearest_fact=Fact(**best),
            failure_mode="wrong_entity",
        ))

    total = len(claims)
    score = verified / total if total else 1.0

    report = AttributionReport(
        mismatches=mismatches,
        verified_claims=verified,
        total_numeric_claims=total,
        attribution_score=score,
    )

    if mismatches:
        wrong_entities = sum(1 for m in mismatches if m.failure_mode == "wrong_entity")
        logger.warning(
            "Attribution verifier — %d/%d claims verified (score=%.2f); "
            "wrong_entity=%d, unit_mismatch=%d, no_supporting_fact=%d",
            verified, total, score, wrong_entities,
            sum(1 for m in mismatches if m.failure_mode == "unit_mismatch"),
            sum(1 for m in mismatches if m.failure_mode == "no_supporting_fact"),
        )
    else:
        logger.info(
            "Attribution verifier — %d/%d claims verified (score=%.2f)",
            verified, total, score,
        )

    return report


def build_rewrite_hint(mismatches: list[dict]) -> str:
    """Build a targeted rewrite hint listing specific (entity, value)
    attribution failures. Consumed by the reasoning agent on retry."""
    if not mismatches:
        return ""
    bullets: list[str] = []
    for m in mismatches[:5]:
        nf = m.get("nearest_fact") or {}
        mode = m.get("failure_mode") or "unknown"
        claim = (m.get("claim_sentence") or "").strip().replace("\n", " ")
        claim = claim[:200] + ("…" if len(claim) > 200 else "")
        if mode == "wrong_entity":
            bullets.append(
                f"- WRONG ENTITY: The sentence \"{claim}\" pairs entity "
                f"\"{m.get('claimed_entity')}\" with value "
                f"\"{m.get('claimed_value_raw')}\". The facts table has this "
                f"value bound to entity \"{nf.get('entity', '?')}\" "
                f"({nf.get('metric', '?')}). EITHER correct the entity to "
                f"match the fact, OR remove the claim."
            )
        elif mode == "unit_mismatch":
            bullets.append(
                f"- UNIT MISMATCH: \"{claim}\" — value "
                f"\"{m.get('claimed_value_raw')}\" should be in unit "
                f"\"{nf.get('unit', '?')}\" per the facts table."
            )
        elif mode == "value_mismatch":
            bullets.append(
                f"- VALUE MISMATCH: \"{claim}\" — facts table has "
                f"\"{nf.get('value_raw', '?')}\" for entity "
                f"\"{nf.get('entity', '?')}\", not \"{m.get('claimed_value_raw')}\"."
            )
        else:  # no_supporting_fact
            bullets.append(
                f"- UNSUPPORTED: \"{claim}\" — no fact in the facts table "
                f"supports \"{m.get('claimed_value_raw')}\" for "
                f"\"{m.get('claimed_entity')}\". Move to Information gaps "
                f"or drop the claim."
            )
    return (
        "ATTRIBUTION ERRORS (DETECTED BY DETERMINISTIC VERIFIER — MUST FIX):\n"
        + "\n".join(bullets)
        + "\n\nUse ONLY values from the FACTS TABLE for numerical claims, "
        "matching entity, value, and unit exactly."
    )
