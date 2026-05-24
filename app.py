"""
FastAPI application — REST API for the Sustainability SME system.
Endpoints: /query, /ingest, /health, and serves the frontend UI.
"""

import json
import logging
import platform
import shutil
import threading
import time
import uuid
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from utils.logging_config import setup_logging

# Initialize logging before anything else
setup_logging(level="INFO")

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PdfAgent",
    description="One-shot agentic RAG over any PDF, with grounded citations.",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Startup pre-warm ────────────────────────────────────────────────────
# Two things need warming so the FIRST real query doesn't pay setup cost:
#   1. BM25 index — singleton lives in-memory only; preprocess builds it,
#      but a fresh uvicorn process starts with an empty BM25, which silently
#      degrades hybrid search to dense-only. Rebuild it from the FAISS
#      docstore on boot.
#   2. Cross-encoder reranker — first call downloads the model from HF Hub
#      (~90 MB). Triggering an import-time load means the user's first query
#      doesn't sit waiting for the download.

@app.on_event("startup")
async def _prewarm() -> None:
    from retrieval.vector_store import vector_store
    from retrieval.bm25_retriever import bm25_index

    # 1. BM25 from FAISS docstore
    try:
        if not bm25_index.is_ready:
            docs = vector_store.get_all_documents()
            if docs:
                bm25_index.build_index(
                    corpus=[d["content"] for d in docs],
                    doc_ids=[d["id"] for d in docs],
                    metadatas=[d.get("metadata", {}) for d in docs],
                )
                logger.info("Startup: BM25 rebuilt from FAISS docstore — docs=%d", len(docs))
            else:
                logger.info("Startup: FAISS docstore empty, skipping BM25 build")
        else:
            logger.info("Startup: BM25 already built (skipped)")
    except Exception as e:
        logger.warning("Startup BM25 rebuild failed: %s", e)

    # 2. Cross-encoder reranker (downloads on first call)
    if settings.rerank_enabled:
        try:
            from retrieval.reranker import _get_reranker
            _get_reranker()
            logger.info("Startup: cross-encoder reranker loaded")
        except Exception as e:
            logger.warning("Startup reranker load failed: %s", e)
    else:
        logger.info("Startup: rerank_enabled=false, skipping cross-encoder load")


# ── Request/Response Models ──

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question")
    thread_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: str = Field(default="default", description="User ID for long-term memory")

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence_score: float
    query_type: str
    query_scope: str
    retrieval_strategy: str
    generated_sql: Optional[str] = None
    sql_results: Optional[list[dict]] = None
    execution_trace: list[dict]
    reasoning_summary: str
    information_gaps: list[str]
    retrieved_chunks: list[dict] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    mentioned_documents: list[str] = Field(default_factory=list)
    token_usage: dict = Field(default_factory=dict)
    faithfulness_score: float = 0.0
    citation_coverage: float = 0.0
    contradictions: list[dict] = Field(default_factory=list)
    arithmetic_corrections: list[dict] = Field(default_factory=list)
    facts: list[dict] = Field(default_factory=list)
    attribution_score: float = 1.0
    attribution_mismatches: list[dict] = Field(default_factory=list)
    thread_id: str
    duration_ms: float

class IngestRequest(BaseModel):
    use_vision: bool = Field(default=True, description="Use vision model for image captioning")
    use_contextual: bool = Field(default=True, description="Apply contextual retrieval enrichment")
    use_raptor: bool = Field(default=True, description="Build RAPTOR hierarchical tree")
    clear_existing: bool = Field(default=False, description="Clear existing data before ingesting")


# ── Endpoints ──

@app.get("/")
async def serve_ui():
    """Serve the frontend UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "UI not found. Place index.html in static/"})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from retrieval.vector_store import vector_store
    from storage.sql_store import sql_store
    return {
        "status": "healthy",
        "vector_store_docs": vector_store.count,
        "sql_tables": len(sql_store.get_table_names()),
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint — runs the full multi-agent pipeline.
    Returns answer, citations, confidence, execution trace, and more.
    """
    from graph.workflow import invoke_query
    from evaluation.response_logger import response_logger

    start = time.time()
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = invoke_query(
            query=request.query,
            thread_id=thread_id,
            user_id=request.user_id,
        )
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    duration = (time.time() - start) * 1000
    payload = _shape_response(result, thread_id, duration)
    if not payload.get("answer"):
        payload["answer"] = "An error occurred while processing your query."
    response = QueryResponse(**payload)

    # Log the response for evaluation
    response_logger.log(request.query, result, duration)

    return response


# Human-readable labels per node — surfaced to the UI as steps complete.
_NODE_LABELS = {
    "query_understanding":     "Understanding your question",
    "memory_read":             "Recalling prior context",
    "planner":                 "Planning the approach",
    "retrieval":               "Searching the reports",
    "table_agent":             "Querying tables",
    "entity_metric_extractor": "Extracting facts",
    "reasoning":               "Synthesizing the answer",
    "validation":              "Verifying citations",
    "memory_save":             "Saving to memory",
}


def _sse(event: str, data: dict) -> str:
    """Serialize an event dict into a Server-Sent-Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming variant of /query. Emits SSE events:
      - event: step      → one per agent that completes
      - event: complete  → final response payload (same shape as /query)
      - event: error     → on failure
    """
    from graph.workflow import (
        workflow,
        _preflight_smalltalk,
        _make_smalltalk_result,
    )
    from langchain_core.messages import HumanMessage
    from evaluation.response_logger import response_logger

    thread_id = request.thread_id or str(uuid.uuid4())

    def event_gen():
        start = time.time()

        # Pre-flight smalltalk — bypass the workflow entirely.
        preflight = _preflight_smalltalk(request.query)
        if preflight is not None:
            yield _sse("step", {
                "node": "preflight",
                "label": "Smalltalk shortcut",
                "status": "complete",
                "summary": "matched greeting — bypassed workflow",
            })
            result = _make_smalltalk_result(request.query, preflight)
            duration = (time.time() - start) * 1000
            payload = _shape_response(result, thread_id, duration)
            response_logger.log(request.query, result, duration)
            yield _sse("complete", payload)
            return

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": request.user_id,
            }
        }
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "original_query": request.query,
            "validation_retries": 0,
            "sql_retries": 0,
            "needs_confirmation": False,
        }

        # Accumulate state across node updates so we can emit a final payload.
        last_state: dict = {}
        try:
            for chunk in workflow.stream(
                initial_state, config=config, stream_mode="updates"
            ):
                # chunk shape: {node_name: state_diff}
                for node_name, state_update in chunk.items():
                    trace = state_update.get("execution_trace", []) if isinstance(state_update, dict) else []
                    summary = ""
                    tokens = {}
                    if trace:
                        last = trace[-1]
                        summary = str(last.get("output_summary", ""))[:160]
                        tokens = last.get("tokens") or {}
                    label = _NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                    event_payload = {
                        "node": node_name,
                        "label": label,
                        "status": "complete",
                        "summary": summary,
                        "tokens": tokens,
                    }
                    # For retrieval, attach the chunks the user can inspect live.
                    if node_name == "retrieval" and isinstance(state_update, dict):
                        event_payload["chunks_preview"] = _shape_chunks(
                            state_update.get("retrieved_chunks", []), limit=10
                        )
                        event_payload["retrieval_queries"] = state_update.get("retrieval_queries", [])
                    yield _sse("step", event_payload)
                    if isinstance(state_update, dict):
                        for k, v in state_update.items():
                            if k == "execution_trace":
                                last_state.setdefault("execution_trace", []).extend(v)
                            else:
                                last_state[k] = v
        except Exception as e:
            logger.exception("Streaming query failed")
            yield _sse("error", {"error": str(e)})
            return

        duration = (time.time() - start) * 1000
        payload = _shape_response(last_state, thread_id, duration)
        response_logger.log(request.query, last_state, duration)
        yield _sse("complete", payload)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable buffering on nginx-style proxies
            "Connection": "keep-alive",
        },
    )


def _shape_response(result: dict, thread_id: str, duration_ms: float) -> dict:
    """Mirror the QueryResponse Pydantic model so the SSE 'complete' payload
    is interchangeable with the JSON returned by /query."""
    trace = result.get("execution_trace", [])
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "confidence_score": result.get("confidence_score", 0.0),
        "query_type": result.get("query_type", "unknown"),
        "query_scope": result.get("query_scope", "unknown"),
        "retrieval_strategy": result.get("retrieval_strategy", "unknown"),
        "generated_sql": result.get("generated_sql"),
        "sql_results": result.get("sql_results") if result.get("sql_results") else None,
        "execution_trace": trace,
        "reasoning_summary": result.get("reasoning_summary", ""),
        "information_gaps": result.get("information_gaps", []),
        "faithfulness_score": result.get("faithfulness_score", 0.0),
        "citation_coverage": result.get("citation_coverage", 0.0),
        "contradictions": result.get("contradictions", []),
        "arithmetic_corrections": result.get("arithmetic_corrections", []),
        "facts": result.get("facts", []),
        "attribution_score": result.get("attribution_score", 1.0),
        "attribution_mismatches": result.get("attribution_mismatches", []),
        "retrieved_chunks": _shape_chunks(result.get("retrieved_chunks", [])),
        "retrieval_queries": result.get("retrieval_queries", []),
        "sub_questions": result.get("sub_questions", []),
        "mentioned_documents": result.get("mentioned_documents", []),
        "token_usage": _aggregate_tokens(trace),
        "thread_id": thread_id,
        "duration_ms": round(duration_ms, 1),
    }


def _aggregate_tokens(trace: list[dict]) -> dict:
    """Sum per-agent token counts into a single per-query total."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    for step in trace or []:
        t = step.get("tokens") or {}
        for k in total:
            try:
                total[k] += int(t.get(k, 0) or 0)
            except (TypeError, ValueError):
                pass
    return total


_TABLE_NAME_RE = __import__("re").compile(r"^[a-zA-Z0-9_]{1,64}$")


def _fetch_table_data(table_name: str, row_limit: int = 100) -> Optional[dict]:
    """Pull the live SQL rows for a table_repr chunk so the UI can render
    them as an actual table (instead of just the LLM-friendly text blob)."""
    if not table_name or not _TABLE_NAME_RE.match(table_name):
        return None
    try:
        from storage.sql_store import sql_store
        df = sql_store.execute_query(f"SELECT * FROM {table_name} LIMIT {row_limit}")
        meta_schema = sql_store.get_schema(table_name)
        return {
            "name": table_name,
            "columns": list(df.columns),
            "column_types": [c.get("type") for c in meta_schema.get("columns", [])],
            "rows": df.where(df.notna(), None).values.tolist(),
            "row_count": int(meta_schema.get("row_count", len(df))),
            "displayed_rows": len(df),
            "source_document": meta_schema.get("source_document"),
            "source_page": meta_schema.get("source_page"),
        }
    except Exception as e:
        logger.debug("Failed to fetch table_data for %s: %s", table_name, e)
        return None


def _shape_chunks(chunks: list[dict], limit: int = 20) -> list[dict]:
    """Truncate retrieved chunks for the API response — keep enough fields for
    the UI to show document/page/type/parent/score, plus a content preview.

    Parents (is_parent_context=True) get a much larger preview so the user can
    actually read what context was fed to the reasoning LLM.
    """
    out = []
    for c in (chunks or [])[:limit]:
        meta = c.get("metadata", {}) or {}
        content = c.get("content", "") or ""
        is_parent = bool(c.get("is_parent_context"))
        # Send (essentially) full content for debugging. Parents top out near
        # ~2k chars, children near ~500; cap at 8k as a safety net for runaway
        # contextualization / unusual chunks.
        preview_cap = 8000

        # If this is a table_repr chunk, fetch the structured rows so the UI
        # can render them as a real <table> instead of the text blob.
        table_data = None
        ctype = meta.get("chunk_type", "child")
        if ctype == "table_repr":
            parent_id = meta.get("parent_id", "") or ""
            if parent_id.startswith("table_"):
                table_data = _fetch_table_data(parent_id[len("table_"):])

        out.append({
            "content_preview": content[:preview_cap] + ("..." if len(content) > preview_cap else ""),
            "table_data": table_data,
            "content_length": len(content),
            "document": meta.get("document_name", "Unknown"),
            "page": meta.get("page_number"),
            "chunk_type": meta.get("chunk_type", "child"),
            "chunk_id": c.get("chunk_id") or meta.get("chunk_id"),
            "parent_id": meta.get("parent_id"),
            "is_parent_context": is_parent,
            "neighbor_expansion": bool(c.get("neighbor_expansion")),
            "raptor_level": meta.get("raptor_level", 0),
            "rerank_score": c.get("rerank_score"),
            "rrf_score": c.get("rrf_score"),
            "dense_score": c.get("dense_score"),
            "bm25_score": c.get("bm25_score"),
        })
    return out


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """Trigger PDF preprocessing pipeline."""
    from preprocessing import preprocess_all_pdfs

    logger.info("Ingestion triggered via API")
    try:
        stats = preprocess_all_pdfs(
            use_vision=request.use_vision,
            use_contextual=request.use_contextual,
            use_raptor=request.use_raptor,
            clear_existing=request.clear_existing,
        )
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error("Ingestion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing."""
    from config.settings import settings

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / file.filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    logger.info("PDF uploaded: %s (%d bytes)", file.filename, len(content))
    return {"status": "uploaded", "filename": file.filename, "size_bytes": len(content)}


@app.get("/schema")
async def get_schema():
    """Get the current database schema catalog."""
    from storage.schema_manager import schema_manager
    return {"catalog": schema_manager.generate_catalog()}


# ── Ollama readiness + single-file upload-ingest ────────────────────────

def _ollama_install_instructions() -> dict:
    """Per-OS instructions the UI shows when Ollama isn't ready."""
    return {
        "download_url": "https://ollama.com/download",
        "windows": [
            "Download Ollama for Windows: https://ollama.com/download/windows",
            "Run the installer. It auto-starts as a tray app.",
            "Open PowerShell and verify: ollama list",
            "Pull required local models: ollama pull mxbai-embed-large",
            "For cloud models (names ending in -cloud) run: ollama signin",
            "Click Retry below once done.",
        ],
        "macos": [
            "Download Ollama for macOS: https://ollama.com/download/mac",
            "Drag Ollama.app to /Applications and launch it.",
            "In a terminal: ollama pull mxbai-embed-large",
            "For cloud models (names ending in -cloud) run: ollama signin",
        ],
        "linux": [
            "curl -fsSL https://ollama.com/install.sh | sh",
            "Start the daemon: ollama serve  (or it auto-starts as a systemd service)",
            "Pull required local models: ollama pull mxbai-embed-large",
            "For cloud models (names ending in -cloud) run: ollama signin",
        ],
    }


def _detect_os_key() -> str:
    sysname = platform.system().lower()
    if sysname == "windows":
        return "windows"
    if sysname == "darwin":
        return "macos"
    return "linux"


def _check_ollama_ready(timeout_s: float = 4.0) -> dict:
    """Probe Ollama and tell the UI exactly what is wrong (and how to fix it).

    Cloud models (names ending in '-cloud') aren't listed by /api/tags so we
    can't verify they're pulled — but reachability + auth is enough to proceed,
    so we surface them as 'cloud_models' rather than 'missing_models'.
    """
    import requests

    required = sorted({
        settings.ollama_model_light,
        settings.ollama_model_heavy,
        settings.ollama_model_vision,
        settings.ollama_model_embed,
    })

    info: dict = {
        "base_url": settings.ollama_base_url,
        "reachable": False,
        "ready": False,
        "models_required": required,
        "models_present": [],
        "models_missing": [],
        "cloud_models": [m for m in required if m.endswith("-cloud")],
        "os": _detect_os_key(),
        "install_instructions": _ollama_install_instructions(),
        "error": None,
    }

    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout_s)
        r.raise_for_status()
        local_models = [m.get("name", "") for m in (r.json() or {}).get("models", [])]
        info["reachable"] = True
        info["models_present"] = local_models

        # Ollama stores tags as "name:tag" and treats an untagged name as
        # ":latest". Normalize both sides so "mxbai-embed-large" matches the
        # listed "mxbai-embed-large:latest".
        def _norm(name: str) -> str:
            return name if ":" in name else f"{name}:latest"
        present_norm = {_norm(m) for m in local_models}
        missing = [
            m for m in required
            if (not m.endswith("-cloud")) and _norm(m) not in present_norm
        ]
        info["models_missing"] = missing
        info["ready"] = (len(missing) == 0)
    except Exception as e:
        info["error"] = str(e)

    return info


@app.get("/ollama/health")
async def ollama_health():
    """Front-end calls this before letting the user upload."""
    return _check_ollama_ready()


def _wipe_all_data() -> dict:
    """Drop FAISS, BM25, SQLite tables, checkpoints, images, traces, PDFs,
    response log. Preserve data/models/ (cross-encoder weights, ~90MB)."""
    from retrieval.vector_store import vector_store
    from retrieval.bm25_retriever import bm25_index
    from storage.sql_store import sql_store
    from storage.schema_manager import schema_manager

    removed = {"in_memory": [], "dirs": [], "files": []}

    for name, fn in (
        ("vector_store", vector_store.clear),
        ("sql_tables", sql_store.drop_all_tables),
        ("bm25", bm25_index.clear),
        ("schema_cache", schema_manager.invalidate_cache),
    ):
        try:
            fn()
            removed["in_memory"].append(name)
        except Exception as e:
            logger.warning("Wipe %s failed: %s", name, e)

    data_dir = Path(settings.data_dir).resolve()
    wipe_dirs = ["faiss", "tables", "checkpoints", "images", "traces", "pdfs"]
    wipe_files = ["response_log.jsonl"]

    for d in wipe_dirs:
        p = data_dir / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            p.mkdir(parents=True, exist_ok=True)
            removed["dirs"].append(d)
    for f in wipe_files:
        p = data_dir / f
        if p.exists():
            p.unlink(missing_ok=True)
            removed["files"].append(f)

    return removed


# A whitelist of preprocessing log records the UI should surface as steps.
# Matching is substring-based and case-sensitive against the formatted message.
_PROGRESS_PATTERNS = [
    ("[1/5] Extracting text",                "Extracting text"),
    ("[2/5] Chunking",                       "Chunking pages"),
    ("[3/5] Extracting tables",              "Extracting tables"),
    ("[4/5] Extracting + OCR",               "Extracting images + OCR + captions"),
    ("[4/5] Skipping image extraction",      "Skipping image extraction"),
    ("[5/5] Contextualizing chunks",         "Contextualizing chunks"),
    ("[5/5] Skipping contextualization",     "Skipping contextualization"),
    ("Indexing",                             "Indexing chunks in FAISS"),
    ("Building BM25 index",                  "Building BM25 index"),
    ("Building RAPTOR hierarchical",         "Building RAPTOR tree"),
    ("Generating schema catalog",            "Generating schema catalog"),
    ("PREPROCESSING COMPLETE",               "Preprocessing complete"),
]


class _QueueLogHandler(logging.Handler):
    """Push selected log records into a queue as SSE 'step' events."""

    def __init__(self, q: "Queue", emit_fn):
        super().__init__(level=logging.INFO)
        self._q = q
        self._emit = emit_fn

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        for needle, label in _PROGRESS_PATTERNS:
            if needle in msg:
                self._emit("step", {"label": label, "status": "complete", "summary": msg[:200]})
                return


def _looks_like_ollama_failure(msg: str) -> bool:
    low = (msg or "").lower()
    return any(t in low for t in (
        "connection refused", "11434", "ollama", "max retries exceeded",
        "failed to establish a new connection", "name or service not known",
    ))


@app.post("/upload-ingest/stream")
async def upload_ingest_stream(file: UploadFile = File(...)):
    """Upload a single PDF, wipe all prior data, run preprocessing.
    Streams SSE 'step' / 'complete' / 'error' events so the UI can show
    progress and surface Ollama install instructions on failure."""
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Read the entire upload before starting the SSE stream — UploadFile is
    # backed by the request body which closes once we return.
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    def event_gen():
        events: "Queue" = Queue()
        done_sentinel = object()

        def emit(event: str, data) -> None:
            events.put((event, data))

        def run_job():
            handler = _QueueLogHandler(events, emit)
            root_logger = logging.getLogger()
            try:
                # 1. Ollama readiness check (early bail-out with install hints)
                emit("step", {"label": "Checking Ollama", "status": "running"})
                ollama = _check_ollama_ready()
                if not ollama["reachable"]:
                    emit("error", {
                        "stage": "ollama",
                        "error": "Ollama is not reachable at " + ollama["base_url"],
                        "ollama": ollama,
                    })
                    return
                if ollama["models_missing"]:
                    emit("error", {
                        "stage": "ollama",
                        "error": "Required models not pulled: " + ", ".join(ollama["models_missing"]),
                        "ollama": ollama,
                    })
                    return
                emit("step", {
                    "label": "Ollama ready",
                    "status": "complete",
                    "summary": f"models: {', '.join(ollama['models_required'])}",
                })

                # 2. Wipe everything except cached model weights
                emit("step", {"label": "Wiping existing data", "status": "running"})
                wiped = _wipe_all_data()
                emit("step", {
                    "label": "Existing data cleared",
                    "status": "complete",
                    "summary": f"cleared {', '.join(wiped['dirs']) or 'nothing'}",
                })

                # 3. Save the uploaded PDF as the sole file in data/pdfs/
                pdf_dir = Path(settings.pdf_dir)
                pdf_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(filename).name  # strip any path traversal
                save_path = pdf_dir / safe_name
                with open(save_path, "wb") as f:
                    f.write(content)
                size_kb = len(content) // 1024
                emit("step", {
                    "label": f"Saved {safe_name}",
                    "status": "complete",
                    "summary": f"{size_kb} KB → {save_path}",
                })

                # 4. Preprocessing — capture log lines as live UI steps.
                root_logger.addHandler(handler)
                emit("step", {"label": "Running preprocessing pipeline", "status": "running"})

                from preprocessing import preprocess_all_pdfs
                stats = preprocess_all_pdfs(
                    use_vision=True,
                    use_contextual=True,
                    use_raptor=True,
                    clear_existing=False,  # already wiped
                )

                emit("complete", {
                    "status": "ready",
                    "filename": safe_name,
                    "stats": stats,
                })

            except Exception as e:
                logger.exception("Upload+ingest failed")
                msg = str(e)
                payload = {"stage": "preprocess", "error": msg}
                if _looks_like_ollama_failure(msg):
                    payload["ollama"] = _check_ollama_ready()
                emit("error", payload)
            finally:
                try:
                    root_logger.removeHandler(handler)
                except Exception:
                    pass
                events.put((done_sentinel, None))

        t = threading.Thread(target=run_job, daemon=True)
        t.start()

        while True:
            try:
                evt, data = events.get(timeout=20)
            except Empty:
                # SSE keepalive comment — prevents proxies from closing the stream
                yield ": keepalive\n\n"
                continue
            if evt is done_sentinel:
                break
            yield _sse(evt, data)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
