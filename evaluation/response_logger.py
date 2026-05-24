"""
Response logging + semantic-similarity helper.

This module is a *library*, not a runnable eval. Two responsibilities:

  - `semantic_similarity(a, b)` — cosine similarity of two strings via
    the active embedding model. Used by ad-hoc analyses and (optionally)
    by tests.

  - `ResponseLogger` — appends a JSONL record per production query to
    `data/response_log.jsonl` for later offline inspection. A
    `response_logger` singleton is imported by `app.py` and called at
    the end of every `/query` and `/query/stream` request.

If you want to *evaluate* the system, use one of:
  - `python -m evaluation.run_all`             (everything below in sequence)
  - `python -m evaluation.agentic_eval`        (end-to-end answer quality)
  - `python -m evaluation.retrieval_eval`      (retrieval-only)
  - `python -m evaluation.adversarial_eval`    (robustness / refusals)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.embeddings import embed_texts

logger = logging.getLogger(__name__)


def semantic_similarity(text1: str, text2: str) -> float:
    """Cosine similarity between two texts using the active embedding model."""
    embeddings = embed_texts([text1, text2])
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ResponseLogger:
    """Appends one JSONL record per production query for offline analysis."""

    def __init__(self, log_path: str = "./data/response_log.jsonl"):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, query: str, result: dict, duration_ms: float) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence_score", 0.0),
            "citations": result.get("citations", []),
            "query_type": result.get("query_type", ""),
            "query_scope": result.get("query_scope", ""),
            "retrieval_strategy": result.get("retrieval_strategy", ""),
            "chunks_retrieved": len(result.get("retrieved_chunks", [])),
            "sql_used": bool(result.get("generated_sql")),
            "validated": result.get("is_validated", False),
            "duration_ms": duration_ms,
            "execution_trace": result.get("execution_trace", []),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.debug("Response logged — query='%s...'", query[:50])

    def get_logs(self, limit: int = 100) -> list[dict]:
        if not Path(self.log_path).exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries[-limit:]


response_logger = ResponseLogger()
