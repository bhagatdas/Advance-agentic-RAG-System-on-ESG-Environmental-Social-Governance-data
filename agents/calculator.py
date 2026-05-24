"""
Calculator — post-pass arithmetic verifier for the Reasoning agent.

The Reasoning prompt asks the LLM to emit a 5-column delta table for every
comparison query:

    | Metric | A | B | Δ absolute | Δ relative |
    |---|---|---|---|---|
    | X     | <val at A> | <val at B> | <B - A> | <(B-A)/A * 100>% |

LLMs frequently get the last two columns wrong on multi-digit ESG values
(e.g. emission tonnes like 1,479,149 vs 1,387,727). This module:

  1. Scans the answer for those delta tables.
  2. Parses A and B from each row.
  3. Recomputes the absolute and relative deltas.
  4. Replaces the LLM's values if they're off by more than a small tolerance.
  5. Returns the corrected answer + a list of corrections applied.

No new dependencies, no LLM call — pure deterministic math.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Scale-word suffixes commonly attached to ESG numbers.
_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6, "mn": 1e6,
    "b": 1e9, "billion": 1e9, "bn": 1e9,
    "t": 1e12, "trillion": 1e12,
}

# Tolerances for treating a claimed delta as "already correct" (covers rounding
# and formatting differences). Absolute floor + relative ceiling: a value
# matches if EITHER bound is satisfied.
_ABS_TOL = 1.0
_REL_TOL = 0.005  # 0.5%


# ── Unit-family map (used to verify A and B share a comparable unit BEFORE
# computing Δ). Each entry: lowercase unit -> (family, factor-to-family-base).
# ESG reports mix scales constantly (kWh / MWh / GWh, t / kt / Mt, tCO2e vs
# tonnes-of-steel) and an LLM that subtracts across units silently produces
# wrong Δ values. We refuse arithmetic across DIFFERENT families and
# normalize across DIFFERENT SCALES within the same family.
#
# Note: tCO2e is intentionally a SEPARATE family from generic mass — they
# happen to share the symbol "t" but conceptually 1 t of steel and 1 tCO2e
# are not addable.
_UNIT_FAMILIES: dict[str, tuple[str, float]] = {
    # Energy (base = Wh)
    "wh":   ("energy", 1.0),
    "kwh":  ("energy", 1e3),
    "mwh":  ("energy", 1e6),
    "gwh":  ("energy", 1e9),
    "twh":  ("energy", 1e12),
    # Power (base = W)
    "w":   ("power", 1.0),
    "kw":  ("power", 1e3),
    "mw":  ("power", 1e6),
    "gw":  ("power", 1e9),
    # Peak (solar) power — distinct family from operational power
    "wp":  ("power_peak", 1.0),
    "kwp": ("power_peak", 1e3),
    "mwp": ("power_peak", 1e6),
    # Mass (base = kg)
    "g":      ("mass", 1e-3),
    "kg":     ("mass", 1.0),
    "tonne":  ("mass", 1e3),
    "tonnes": ("mass", 1e3),
    "ton":    ("mass", 1e3),
    "tons":   ("mass", 1e3),
    "kt":     ("mass", 1e6),
    # Note: bare "mt" / "gt" / "t" omitted — too ambiguous with scale words.
    # The LLM-side prompt asks for explicit unit names, and ESG reports
    # typically write "tonnes" / "kt" / "Mt CO2e" in full.
    # CO2 / GHG emissions (base = kgCO2e). SEPARATE from generic mass.
    "kgco2e":  ("co2e", 1.0),
    "tco2e":   ("co2e", 1e3),
    "ktco2e":  ("co2e", 1e6),
    "mtco2e":  ("co2e", 1e9),
    "tco2":    ("co2e", 1e3),
    # Volume (base = L)
    "l":  ("volume", 1.0),
    "ml": ("volume", 1e-3),    # milliliter
    "kl": ("volume", 1e3),
    "m3": ("volume", 1e3),
    # Area (base = m²)
    "m2":  ("area", 1.0),
    "km2": ("area", 1e6),
    "ha":  ("area", 1e4),
    # Percent (dimensionless)
    "%":       ("percent", 1.0),
    "pct":     ("percent", 1.0),
    "percent": ("percent", 1.0),
}

# Matches signed/unsigned numbers with optional commas and decimals.
_NUMBER_RE = re.compile(
    r"""
    [+-]?
    (?:\d{1,3}(?:,\d{3})+|\d+)
    (?:\.\d+)?
    """,
    re.VERBOSE,
)


@dataclass
class Correction:
    """One finding from the deterministic post-pass on the answer.

    `kind` distinguishes:
      - "value"              — the LLM's Δ was numerically wrong; replaced.
      - "unit_normalization" — A and B had different scales of the same unit
        (e.g. kWh vs MWh); we converted to a common scale before computing Δ.
        The row's Δ values were ALSO replaced if they used the LLM's bad math.
      - "unit_incompatible"  — A and B had units from different families
        (e.g. kWh vs tCO2e); arithmetic was SKIPPED for this row and the
        LLM's values were left untouched (we can't verify them).
    """
    row: str           # the metric / row label
    column: str        # header cell that was corrected (e.g. "Δ absolute")
    llm_value: str     # what the LLM wrote
    computed_value: str  # what the math actually is
    kind: str = "value"
    note: str = ""


@dataclass
class CalcResult:
    corrected_answer: str
    corrections: list[Correction] = field(default_factory=list)


def _extract_unit(s: str) -> str:
    """Pull the unit suffix off an ESG value string.

    "1278 kWp"          → "kWp"
    "1.8 million kWh"   → "kWh"  (scale word skipped)
    "24%"               → "%"
    "$9.5M"             → ""     (currency-style, no unit family we model)
    "1,479,149 tCO2e"   → "tCO2e"
    "—" / ""            → ""

    Case is preserved in the returned string but `_unit_family` lower-cases it
    for the lookup — keeps the display form intact for user-visible notes.
    """
    if not s:
        return ""
    t = s.strip()
    if not t or t in {"—", "-", "N/A", "n/a", "TBD", "TBA"}:
        return ""
    m = _NUMBER_RE.search(t)
    if not m:
        return ""
    tail = t[m.end():].strip()
    if not tail:
        return ""
    # Skip a scale word if present at the start of the tail
    parts = tail.split(maxsplit=1)
    if parts and parts[0].lower() in _SCALE:
        tail = parts[1] if len(parts) > 1 else ""
    # Strip trailing punctuation/whitespace
    return tail.strip(" .,;:)]")


def _unit_family(unit: str) -> tuple[str | None, float | None]:
    """Look up (family, factor-to-base) for a unit. Returns (None, None) if
    we don't know the unit — caller should fall back to assuming same unit."""
    if not unit:
        return None, None
    info = _UNIT_FAMILIES.get(unit.strip().lower())
    if info is None:
        return None, None
    return info


def _check_unit_pair(a_str: str, b_str: str) -> tuple[str, str | None, float | None, float | None, str]:
    """Inspect the units on two value cells.

    Returns a tuple (status, family, a_factor_to_base, b_factor_to_base, note).
    status ∈ {
        "match",          — same unit OR both unknown; safe to proceed
        "normalize",      — same family, different scales; convert B to A
        "incompatible",   — different families; arithmetic is meaningless
    }
    a_factor / b_factor are populated only when status == "normalize". They
    express how to convert each cell to a common base (a_base = a * a_factor).
    """
    a_unit = _extract_unit(a_str)
    b_unit = _extract_unit(b_str)
    a_fam, a_fac = _unit_family(a_unit)
    b_fam, b_fac = _unit_family(b_unit)

    # Both unknown OR exactly identical → assume same unit (existing behavior)
    if a_unit.lower() == b_unit.lower():
        return "match", a_fam, None, None, ""

    # One side has a known unit, the other doesn't — be forgiving, treat
    # as same. Most ESG delta tables only put the unit on the column header
    # or on one of the cells.
    if a_fam is None or b_fam is None:
        return "match", a_fam or b_fam, None, None, ""

    if a_fam != b_fam:
        note = (
            f"A is {a_fam} ({a_unit!s}), B is {b_fam} ({b_unit!s}) — different unit families. "
            f"Arithmetic skipped: A and B are not comparable."
        )
        return "incompatible", None, None, None, note

    # Same family, different scales — convert both to a common base, then
    # back to A's unit for the output.
    note = (
        f"Aligned B from {b_unit} to {a_unit} ({a_fam}) before computing Δ "
        f"(factor B/A = {b_fac / a_fac:g})."
    )
    return "normalize", a_fam, a_fac, b_fac, note


def _parse_number(s: str) -> float | None:
    """Best-effort parse of an ESG-style number with optional unit suffix.

    "1,479,149"     → 1479149.0
    "$9.5M"         → 9500000.0
    "8.14%"         → 8.14
    "—" / "N/A"     → None
    """
    if not s:
        return None
    t = s.strip()
    if not t or t in {"—", "-", "N/A", "n/a", "TBD", "TBA"}:
        return None

    m = _NUMBER_RE.search(t)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None

    # Look for a scale word right after the number (e.g. "9.5M", "9.5 million").
    tail = t[m.end():].strip().lower()
    if tail:
        token = re.split(r"[\s%]", tail, maxsplit=1)[0]
        scale = _SCALE.get(token)
        if scale:
            value *= scale

    return value


def _format_like(template: str, value: float) -> str:
    """Format `value` using formatting cues from `template` (commas, %, decimals)."""
    has_pct = "%" in template
    has_comma = "," in template
    m = re.search(r"\.(\d+)", template)
    decimals = len(m.group(1)) if m else (2 if has_pct else 0)

    abs_v = abs(value)
    if has_comma and abs_v >= 1000:
        formatted = f"{value:,.{decimals}f}"
    else:
        formatted = f"{value:.{decimals}f}"
    if has_pct:
        formatted += "%"
    return formatted


def _is_close(claimed: float, computed: float) -> bool:
    if abs(claimed - computed) <= _ABS_TOL:
        return True
    if computed == 0:
        return False
    return abs(claimed - computed) / abs(computed) <= _REL_TOL


def _looks_like_delta_table(header_cells: list[str]) -> bool:
    """True for 5-column tables with Δ / delta / change in the last two headers."""
    if len(header_cells) != 5:
        return False
    last_two = " ".join(header_cells[-2:]).lower()
    if "δ" in last_two or "delta" in last_two or "change" in last_two:
        return True
    # Also accept "absolute" + "relative" naming.
    return "absolute" in last_two and "relative" in last_two


def verify_arithmetic(answer: str) -> CalcResult:
    """
    Find delta-style markdown tables in the answer and verify the Δ columns.
    Returns the (possibly modified) answer plus a list of corrections.
    """
    if not answer or "|" not in answer:
        return CalcResult(corrected_answer=answer)

    corrections: list[Correction] = []
    lines = answer.split("\n")
    out_lines = list(lines)

    i = 0
    while i < len(lines) - 2:
        header = lines[i]
        if "|" not in header:
            i += 1
            continue
        divider = lines[i + 1].strip()
        if not re.match(r"^\|?[\s:\-|]+\|?$", divider) or "-" not in divider:
            i += 1
            continue

        header_cells = [c.strip() for c in header.strip().strip("|").split("|")]
        if not _looks_like_delta_table(header_cells):
            i += 1
            continue

        # Iterate body rows until we hit a non-table line.
        j = i + 2
        while j < len(lines):
            row = lines[j]
            if "|" not in row or row.strip() == "":
                break
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) != 5:
                break

            metric, a_str, b_str, abs_str, rel_str = cells
            a = _parse_number(a_str)
            b = _parse_number(b_str)
            if a is None or b is None:
                j += 1
                continue

            # ── Unit-consistency pre-check ─────────────────────────────────
            # Runs BEFORE arithmetic so the calculator never subtracts across
            # mismatched units (kWh − MWh, t − tCO2e). Three outcomes:
            #   - "match":         proceed as before
            #   - "normalize":     same family, different scales — convert B
            #                      into A's unit, then compute Δ in A's unit
            #   - "incompatible":  different families — SKIP value verification
            #                      for this row, just surface the finding
            status, _fam, a_factor, b_factor, unit_note = _check_unit_pair(a_str, b_str)

            if status == "incompatible":
                corrections.append(Correction(
                    row=metric,
                    column="(unit check)",
                    llm_value=f"A={a_str} ⟂ B={b_str}",
                    computed_value="(arithmetic skipped)",
                    kind="unit_incompatible",
                    note=unit_note,
                ))
                # Do not overwrite the LLM's Δ values — we can't validate them
                # without a unit conversion we don't have.
                j += 1
                continue

            if status == "normalize":
                # Convert both A and B to a common base, then express Δ in
                # the SMALLER of the two scales — that's the unit the LLM
                # almost certainly worked in (Δ "980 kWh" vs Δ "0.98 MWh"
                # — the former is integer-clean). We also use that side's
                # cell as the format template so commas/decimals match the
                # finer granularity.
                af = a_factor or 1.0
                bf = b_factor or 1.0
                a_base = a * af
                b_base = b * bf
                abs_base = b_base - a_base
                target_factor = min(af, bf)  # smaller scale = finer granularity
                abs_computed = abs_base / target_factor
                rel_computed = (abs_base / a_base) * 100 if a_base != 0 else None
                # Remember which cell to use as the format template downstream
                template_str_for_abs = a_str if af <= bf else b_str
                corrections.append(Correction(
                    row=metric,
                    column="(unit normalization)",
                    llm_value=f"A={a_str}, B={b_str}",
                    computed_value=f"computed Δ in {'A' if af <= bf else 'B'}'s unit (smaller scale)",
                    kind="unit_normalization",
                    note=unit_note,
                ))
            else:  # "match"
                abs_computed = b - a
                rel_computed = (b - a) / a * 100 if a != 0 else None
                template_str_for_abs = abs_str

            new_cells = list(cells)
            row_changed = False

            abs_claimed = _parse_number(abs_str)
            if abs_claimed is not None and not _is_close(abs_claimed, abs_computed):
                # When the LLM's Δ template carries the SAME unit as the
                # template we picked above, use the LLM's template directly so
                # commas/decimals match what they wrote. Otherwise fall back
                # to the smaller-scale operand's template.
                fmt_template = abs_str if _extract_unit(abs_str) == _extract_unit(template_str_for_abs) else template_str_for_abs
                new_cells[3] = _format_like(fmt_template, abs_computed)
                corrections.append(Correction(
                    row=metric, column=header_cells[3],
                    llm_value=abs_str, computed_value=new_cells[3],
                    kind="value",
                ))
                row_changed = True

            if rel_computed is not None:
                rel_claimed = _parse_number(rel_str)
                if rel_claimed is not None and not _is_close(rel_claimed, rel_computed):
                    new_cells[4] = _format_like(rel_str, rel_computed)
                    corrections.append(Correction(
                        row=metric, column=header_cells[4],
                        llm_value=rel_str, computed_value=new_cells[4],
                        kind="value",
                    ))
                    row_changed = True

            if row_changed:
                out_lines[j] = "| " + " | ".join(new_cells) + " |"
            j += 1

        i = j  # skip past this table

    return CalcResult(
        corrected_answer="\n".join(out_lines),
        corrections=corrections,
    )
