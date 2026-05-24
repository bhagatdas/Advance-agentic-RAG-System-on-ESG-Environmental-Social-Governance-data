"""
Local tracing — drop-in replacement for `langsmith.traceable`.

Every agent node uses `@traceable(run_type="chain", name="...")`. This module
provides the same decorator API but logs runs to a local JSONL file under
`data/traces/` instead of pinging LangSmith's cloud. That way:

  - No 429s, no monthly trace quotas, no network dependency.
  - The existing `execution_trace` field on `AgentState` is unchanged; this
    decorator is independent of it and gives you a per-node call log even
    when the workflow short-circuits before populating `execution_trace`.

Behavior is controlled by `LOCAL_TRACING` (env var or `.env`):

  LOCAL_TRACING=true   (default) — append one JSON line per call to
                                    `data/traces/trace_<YYYY-MM-DD>.jsonl`
  LOCAL_TRACING=false             — pure no-op (still runs the function;
                                    nothing is recorded)

Each record:
  {
    "ts":         ISO timestamp,
    "run_id":     8-char hex,
    "name":       node name (e.g. "Planner"),
    "run_type":   "chain" / "tool" / "llm" / ...,
    "elapsed_ms": float,
    "error":      stringified exception, or null on success
  }

Reading the file back is just `cat data/traces/trace_*.jsonl` or jq.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _is_enabled() -> bool:
    return os.getenv("LOCAL_TRACING", "true").strip().lower() in ("1", "true", "yes", "on")


def _trace_dir() -> Path:
    # Resolved at write time so it picks up settings.data_dir overrides cleanly.
    d = Path(os.getenv("LOCAL_TRACE_DIR", "data/traces"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trace_file() -> Path:
    return _trace_dir() / f"trace_{time.strftime('%Y-%m-%d')}.jsonl"


def _write_record(record: dict) -> None:
    """Append one JSON line. Best-effort — never raises into the caller."""
    try:
        with open(_trace_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        # Logging the failure is enough — tracing must never break the pipeline.
        logger.debug("local trace write failed: %s", e)


def traceable(run_type: str = "chain", name: str | None = None) -> Callable:
    """
    Local stand-in for `langsmith.traceable`. Accepts the same kwargs the
    codebase uses; ignores anything else.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        node_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _is_enabled():
                return func(*args, **kwargs)

            run_id = uuid.uuid4().hex[:8]
            start = time.time()
            error: str | None = None
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                _write_record({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "run_id": run_id,
                    "name": node_name,
                    "run_type": run_type,
                    "elapsed_ms": round((time.time() - start) * 1000, 1),
                    "error": error,
                })

        return wrapper

    return decorator
