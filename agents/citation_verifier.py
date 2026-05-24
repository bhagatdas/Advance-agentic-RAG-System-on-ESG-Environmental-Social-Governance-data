"""
Citation verifier — ensures every claim the Reasoning agent cites is actually
present in the retrieved context.

Two checks run side-by-side:

1. Structured citations ({document, page, chunk_id}) from `ReasonedAnswer.citations`
   are matched against the metadata of retrieved chunks. A citation is verified
   when at least one retrieved chunk shares the same (document_name, page_number)
   — with document name compared by case-insensitive substring on either side so
   "hon-esg-report" matches "hon-esg-report-25-53.pdf".

2. Inline citations parsed out of the answer text — patterns like
   "[document_name, Page 42]" — are also checked the same way. The reasoning
   prompt asks for these explicitly, so failing to match them is a strong signal
   of hallucination.

If either coverage drops below the configured threshold, the validator forces a
"rewrite_answer" verdict so the CRAG loop gets a chance to fix it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from config.settings import settings

logger = logging.getLogger(__name__)

# Matches "[Document, Page 12]" or "[document.pdf, Page 12]" or
# "[Document, p. 12]" / "[Document, p12]" — case-insensitive.
_INLINE_CITATION_RE = re.compile(
    r"\[([^\[\]]+?),\s*(?:page|p\.?)\s*(\d+)\]",
    flags=re.IGNORECASE,
)


@dataclass
class CitationCheck:
    """Result of verifying a single citation against retrieved chunks."""
    document: str
    page: int
    source: str  # "structured" | "inline"
    verified: bool
    matched_chunk_id: str = ""


@dataclass
class VerificationReport:
    verified: list[CitationCheck]
    unverified: list[CitationCheck]
    coverage: float           # verified / (verified + unverified), 1.0 when no citations
    structured_total: int
    inline_total: int

    def to_state_dict(self) -> dict:
        return {
            "verified_citations": [c.__dict__ for c in self.verified],
            "unverified_citations": [c.__dict__ for c in self.unverified],
            "citation_coverage": self.coverage,
        }


def _normalize_doc(name: str) -> str:
    """Lowercase + strip extension + collapse whitespace for fuzzy doc matching."""
    if not name:
        return ""
    n = name.strip().lower()
    # Strip a trailing .pdf if the LLM included the extension
    if n.endswith(".pdf"):
        n = n[:-4]
    return n


def _docs_match(cited: str, chunk_doc: str) -> bool:
    """A citation's document matches a chunk's document_name when either string
    is contained in the other (after normalization)."""
    a = _normalize_doc(cited)
    b = _normalize_doc(chunk_doc)
    if not a or not b:
        return False
    return a in b or b in a


def _pages_match(cited_page: int, chunk_page) -> bool:
    try:
        return int(cited_page) == int(chunk_page)
    except (TypeError, ValueError):
        return False


def _find_supporting_chunk(
    cited_doc: str,
    cited_page: int,
    retrieved_chunks: list[dict],
) -> str:
    """Return the id of any retrieved chunk matching (doc, page), else ''."""
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {}) or {}
        if _docs_match(cited_doc, meta.get("document_name", "")) and _pages_match(
            cited_page, meta.get("page_number", -1)
        ):
            return chunk.get("id", "") or chunk.get("chunk_id", "")
    return ""


def _parse_inline_citations(answer: str) -> list[tuple[str, int]]:
    """Extract (document, page) pairs from inline [Doc, Page N] markers."""
    pairs: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for match in _INLINE_CITATION_RE.finditer(answer or ""):
        doc = match.group(1).strip()
        try:
            page = int(match.group(2))
        except ValueError:
            continue
        # Skip the canned refusal "[topic]" placeholder from the reasoning prompt
        if doc.lower() in {"topic", "document", "documentname"}:
            continue
        key = (_normalize_doc(doc), page)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((doc, page))
    return pairs


def verify_citations(
    citations: list[dict],
    answer: str,
    retrieved_chunks: list[dict],
) -> VerificationReport:
    """
    Verify both structured `Citation` objects and inline [Doc, Page N] markers
    against the retrieved chunks that the Reasoning agent was given.

    A "verified" citation has at least one retrieved chunk with a matching
    document_name (fuzzy) and page_number (exact).
    """
    verified: list[CitationCheck] = []
    unverified: list[CitationCheck] = []

    # 1. Structured citations from the Pydantic model
    for cit in citations or []:
        doc = cit.get("document", "")
        try:
            page = int(cit.get("page", -1))
        except (TypeError, ValueError):
            page = -1

        match_id = _find_supporting_chunk(doc, page, retrieved_chunks)
        check = CitationCheck(
            document=doc, page=page, source="structured",
            verified=bool(match_id), matched_chunk_id=match_id,
        )
        (verified if check.verified else unverified).append(check)

    # 2. Inline citation markers in the answer text
    structured_keys = {(_normalize_doc(c.document), c.page) for c in verified + unverified}
    for doc, page in _parse_inline_citations(answer):
        key = (_normalize_doc(doc), page)
        if key in structured_keys:
            # Already covered by a structured citation — don't double-count
            continue
        match_id = _find_supporting_chunk(doc, page, retrieved_chunks)
        check = CitationCheck(
            document=doc, page=page, source="inline",
            verified=bool(match_id), matched_chunk_id=match_id,
        )
        (verified if check.verified else unverified).append(check)

    total = len(verified) + len(unverified)
    # No citations at all → coverage is undefined; treat as 1.0 so we don't
    # punish answers that legitimately have nothing to cite (refusals).
    coverage = (len(verified) / total) if total else 1.0

    report = VerificationReport(
        verified=verified,
        unverified=unverified,
        coverage=coverage,
        structured_total=len(citations or []),
        inline_total=len(_parse_inline_citations(answer)),
    )
    logger.info(
        "Citation verification — total=%d verified=%d unverified=%d coverage=%.2f",
        total, len(verified), len(unverified), coverage,
    )
    return report


def coverage_below_threshold(coverage: float) -> bool:
    """True when the answer should be rewritten because too few cites match."""
    return coverage < settings.citation_coverage_threshold
