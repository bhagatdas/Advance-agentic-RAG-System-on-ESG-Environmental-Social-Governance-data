# ESG Insight Pro by Bhagat Labs — Complete System Specification

> **Purpose**: This document captures the full end-to-end design of the ESG Insight Pro system.
> Use this as a prompt, reference, or onboarding guide for understanding or reproducing the system.

---

## What This System Does

An AI-powered **Subject Matter Expert** for ESG (Environmental, Social, Governance) that answers complex sustainability queries from large enterprise PDF reports (50+ pages, multimodal: text, tables, images). It is NOT a simple chatbot — it is a **stateful, multi-agent, goal-driven reasoning system**.

---

## Architecture Overview

```
User Query
  |
  v
[Pre-flight Smalltalk Shortcut] -- rule-based; obvious greetings return canned reply (0 LLM calls)
  |
  v
[Query Understanding Agent] -- classifies type/scope/intent, extracts entities + mentioned_documents
  |
  v
[Memory Agent (Read)] -- retrieves relevant past interactions from LangGraph Store
  |
  v
[Planner Agent] -- decides: (1) end-early? (2) RAG? (3) Table? (4) strategy
                   also emits 2-4 sub_questions and a complexity_tier
                   (trivial / moderate / complex) that gates expensive nodes
  |
  +---> [Retrieval Agent] -- hybrid search (semantic + BM25) + (optional) reranking
  |       |                    + per-sub-question fan-out (if decomposed)
  |       |                    + per-source diversity cap on the top-K
  |       |                    + metadata pre-filter (mentioned documents)
  |       |                    + neighbor-page parent expansion
  |       |                    OR RAPTOR tree search (for global queries)
  |       |                    OR Map-Reduce scan (for aggregation queries)
  |       v
  +---> [Table Agent] -- generates SQL from natural language, executes against SQLite
  |       |
  |       v
  +-> [Entity-Metric Extractor] -- pre-synthesis fact extraction; emits atomic
                                   (entity, metric, value, unit, source) tuples
                                   from chunks + SQL rows. Skipped on
                                   complexity_tier="trivial".
              |
              v
        [Reasoning Agent] -- sources EVERY numeric claim from the facts table;
                             adaptive markdown (lead paragraph + organic headers,
                             optional **Bottom line:**, required delta tables)
                                  -> [Citation Verifier]      (extractive doc/page check)
                                  -> [Contradiction Detector] (cross-chunk numeric / factual conflict scan)
                                  -> [Calculator]             (deterministic arithmetic verifier for delta tables)
                                  -> [Attribution Verifier]   (deterministic entity↔value check vs. facts table)
              |
              v
            [Validation Agent (CRAG + hard overrides)] -- grounding verdict AND
                                                          claim-level faithfulness
                                                          in ONE LLM call
              |
              +--- PASS --> [Memory Agent (Save)] --> Response to User
              +--- FAIL --> loops back to Retrieval or Reasoning (max 2 retries);
                            attribution failures fed back as a targeted rewrite_hint
```

The whole pipeline is exposed via these endpoints:
- `POST /query`                — synchronous JSON
- `POST /query/stream`         — SSE stream of one event per agent completion, so the UI can show live progress
- `POST /upload-ingest/stream` — single-PDF upload + automatic wipe-then-ingest with SSE-streamed pipeline progress; what the **Upload PDF** button in the browser UI uses
- `GET  /ollama/health`        — probes Ollama reachability + checks required models are pulled; returns per-OS install instructions and flags `-cloud` models that need `ollama signin`. Used by the UI to pre-flight before letting the user upload, and to surface in-app install help when ingestion fails on a connection error.

---

## Agents (core + verifiers)

| Agent | Model | Purpose |
|-------|-------|---------|
| **Query Understanding** | Light (gpt-oss:120b-cloud) | Classify query type/scope/intent, rewrite for clarity, extract entities + mentioned documents |
| **Memory Read** | None (LangGraph Store) | Retrieve relevant past interactions for the user |
| **Planner** | Light | Decide end-early / RAG / Table / Both + strategy + sub-questions + complexity_tier (trivial/moderate/complex) |
| **Retrieval** | Light (for query rewriting) | Multi-query / per-sub-question hybrid search + (optional) cross-encoder rerank + parent expansion + neighbor-page expansion + metadata pre-filter on mentioned documents + per-source diversity cap |
| **Table** | Light | Generate SQLite SELECT queries from natural language using schema catalog |
| **Entity-Metric Extractor** | Light (+ deterministic SQL path) | Pre-synthesis fact extraction: emit atomic (entity, metric, value, unit, source_chunk_id) tuples used as the authoritative source for all numeric claims downstream. Skipped on complexity_tier="trivial" or when attribution_enabled=false |
| **Reasoning** | Heavy (gpt-oss:120b-cloud) | Synthesize answer using adaptive markdown (lead paragraph + organic section headers; optional Bottom-line callout; required 5-column delta table for YoY comparisons); sources every number from the facts table; cites every claim |
| **Citation Verifier** | None (extractive) | Verify every `[Doc, Page N]` marker resolves to an actually-retrieved chunk |
| **Attribution Verifier** | None (deterministic) | For each numeric mention in the answer, look up the (entity, value, unit) triple in the facts table. Classify failures as wrong_entity / no_supporting_fact / unit_mismatch / value_mismatch. Score < threshold → forced rewrite |
| **Contradiction Detector** | Light | Scan retrieved chunks for numeric / factual conflicts about the same metric, year, scope — chunks vs. each other (not answer vs. chunks) |
| **Calculator** | None (deterministic) | Parse delta tables in the answer, recompute Δ absolute / Δ relative, replace LLM values that differ beyond tolerance |
| **Validation** | Light | CRAG grounding verdict + claim-level faithfulness decomposition in ONE LLM call (previously two); hard overrides on citation coverage, faithfulness, and attribution; can emit a targeted rewrite_hint for the next reasoning pass |
| **Memory Save** | Light | Summarize the Q&A for long-term memory |

---

## Key Techniques

### Advanced RAG
- **Parent-child chunking**: Small children for precise search, large parents for context
- **Hybrid retrieval**: Semantic (FAISS / IndexFlatIP cosine) + keyword (BM25) merged via Reciprocal Rank Fusion
- **Optional cross-encoder reranking**: Precision filter using `ms-marco-MiniLM-L-6-v2`. Toggleable via `rerank_enabled` (default off — falls back to top-K of hybrid RRF when disabled)
- **Per-source diversity cap**: `rerank_per_source_max` (default 2) limits how many chunks from the same (document, page) survive the top-K so a long narrative section can't crowd out other entities. Aggregation queries get a wider `rerank_top_k_aggregation` budget
- **Neighbor-page parent expansion**: When a top-K reranked child sits next to a page that hybrid search also surfaced, pull that page's parents into context
- **Multi-query rewriting**: Generate 3 query variants for better recall
- **Query decomposition**: For compound / multi-hop queries, the planner emits 2-4 sub-questions, retrieval fans out per sub-question, and the Reasoning agent receives a **per-sub-question scoped evidence** block instead of one merged pool — mimics what extended-thinking models do implicitly
- **Metadata-aware retrieval**: `mentioned_documents` (extracted by Query Understanding) pre-filters the candidate pool before reranking, with graceful fallback when the filter zeros out results
- **Context compression**: Only top-K reranked chunks sent to LLM
- **BM25 startup pre-warm**: rebuild from FAISS docstore on uvicorn boot so a fresh process never silently degrades to dense-only

### Solving the Harry Potter Problem (Global vs Local Queries)
- **RAPTOR Tree**: Hierarchical summaries (chunk clusters → section summaries → document overview)
  - Local queries → search Level 0 (raw chunks)
  - Global queries → search Level 2-3 (section/document summaries)
- **Map-Reduce**: For exhaustive aggregation queries ("list ALL X"), scan every chunk with focused extraction prompt, then merge/deduplicate
- **Adaptive routing**: Query Understanding classifies scope → Planner picks strategy

### Contextual Retrieval (Anthropic)
- Each chunk gets a 2-3 sentence context prefix during preprocessing
- Reduces retrieval failures by ~49%

### Dynamic Context-Window Management
- `utils/llm._detect_model_context_length` queries Ollama `/api/show` once per process per model and reads `<family>.context_length` (gpt-oss reports `131072`)
- Clamped to `MAX_CONTEXT_TOKENS` and passed to `ChatOllama` as `num_ctx` — Ollama otherwise defaults `num_ctx=2048` regardless of model capacity, silently truncating large prompts
- `utils/llm.get_context_budgets()` computes per-slot **character** caps as **percentages** of `(num_ctx − OUTPUT_TOKENS_RESERVED − PROMPT_OVERHEAD_TOKENS) × CHARS_PER_TOKEN`. Shares are tunable per slot (`CTX_SHARE_TEXT/TABLE/MAP_REDUCE/MEMORY`)
- The Reasoning node consumes these budgets instead of hardcoded char caps, so swapping to a 16K-context model auto-shrinks `text_context` instead of overflowing

### Corrective RAG (CRAG) + Hallucination Defense
- Validation agent can: pass, rewrite answer, or re-retrieve with modified query — in ONE LLM call that also performs claim-level faithfulness decomposition (previously a separate faithfulness_checker.py, merged for ~1/2 the token spend)
- Three **hard overrides** on top of CRAG verdicts:
  - Citation coverage below threshold (default 0.5) → forced `rewrite_answer`
  - Faithfulness score below threshold (default 0.7) → forced `rewrite_answer`
  - Attribution score below threshold (default 0.9) → forced `rewrite_answer` with a targeted `rewrite_hint` listing the specific (entity, value) failures
- Max 2 retry loops before giving up with low confidence
- `rewrite_hint` is consumed by the next reasoning pass and cleared after use

### Pre-synthesis Fact Extraction (the cross-row contamination fix)
- Failure mode this addresses: when retrieved chunks contain a list of entities each with their own numbers (e.g. multiple solar sites with capacity / generation / offset values), the Reasoning LLM frequently mixes (entity, value) pairs across rows in its prose. Existing verifiers can't catch it — citation_verifier checks (doc, page) only, the claim-level judge sees both names and both numbers in the context, and the contradiction detector compares chunks to chunks (not answer to chunks).
- The `entity_metric_extractor` node runs AFTER retrieval/table and BEFORE reasoning. SQL rows convert to facts deterministically (no LLM); retrieved chunks go through one LIGHT LLM call. Chunks are prefixed with unique `[CHUNK_<id>]` markers so each fact cites its source span; facts citing unknown chunks are dropped.
- The Reasoning prompt renders the facts table verbatim and is told to source ALL numeric claims from it. List-shaped passages get a TABLE format (one row per entity) instead of a comma-separated prose list — preserving the (entity → value) pairing.
- The deterministic `attribution_verifier` then checks the final answer against the same facts table after generation, with no LLM call.

### Structured Output
- Every agent returns a Pydantic model (`QueryAnalysis`, `ExecutionPlan`, `SQLGeneration`, `ReasonedAnswer`, `ValidationResult` — now carrying merged `claim_verdicts`, `Fact`, `FactsTable`, `AttributionMismatch`, `AttributionReport`, `Contradiction`, `ContradictionReport`)
- Type-safe, parseable, validated at every step

### Adaptive Answer Format
- Lead paragraph (1-3 sentences) stating the answer directly, no fixed prefix
- Organic section headers (`##` / `###`) chosen to fit the actual content — pick from "Key findings / Drivers / Methodology / Recent results / Outstanding questions" or domain-specific groupings the evidence naturally implies (by region, by product line, by time period)
- Within each section: bulleted lists with a **bolded entity name** at the start of each bullet for quick scanning, citation `[Document, Page X]` at the end of every claim
- Optional `**Bottom line:**` callout — ONLY when the question has a clean one-line answer (single number, yes/no, single named entity)
- Optional Evidence-by-category table — only when categorization adds clarity AND bullets don't already convey it
- Required 5-column delta table for YoY / multi-period comparisons: `| Metric | A | B | Δ absolute | Δ relative |`. The calculator verifies the Δ columns afterwards
- Information gaps — bulleted list of what the reports do NOT say; for compound queries, lists which specific parts had no supporting data
- Refusals stay verbatim (no formatting applied)
- **Compound query rule**: when the user asks multiple things, answer the parts whose data IS in the context and list the unanswerable parts under Information gaps — only emit the full refusal string when NONE of the parts can be answered

### SSE Streaming
- `POST /query/stream` emits one `event: step` per agent completion (`{node, label, summary}`) and a final `event: complete` with the full response payload
- Frontend renders a live checklist that fills in as agents finish, drastically reducing perceived latency on 30-60s queries

---

## Preprocessing Pipeline (Offline)

```
PDF → PyMuPDF Parse
  ├── Text per page → Parent-child chunking → Contextual enrichment → FAISS + BM25
  ├── Tables per page → DataFrame → merged-cell splitter → SQLite + Schema catalog + Text representation → FAISS
  ├── Images per page → EasyOCR + Vision-LLM captioning → Text chunks → FAISS  (optional, --no-images skips)
  └── All chunks → RAPTOR tree builder → Hierarchical summaries → FAISS
```

Single entry point is `preprocessing.py` (formerly an `ingestion/` package).

**Per-page metadata fix**: pymupdf4llm's per-chunk metadata uses the key `page_number` (1-indexed), not `page`. The previous extractor read `.get("page", 0) + 1` and silently labelled every chunk page=1, which propagated into `parent_id = <doc>_p1_parent0` for every parent and broke `[Doc, Page N]` citations. Fixed in `_extract_with_pymupdf4llm` — re-ingest after pulling.

Flags:
- `--clear` — wipe FAISS + SQLite + BM25 first
- `--no-vision` — skip vision-LLM captions (faster)
- `--no-images` — skip image extraction entirely (fastest)
- `--no-contextual` — skip Anthropic-style context prefixes
- `--no-raptor` — skip RAPTOR tree build
- `--no-ocr-fallback` — disable EasyOCR fallback for pages where PyMuPDF returns little text

### Table Storage (SQL)
- All PDF tables stored in SQLite with auto-generated schema catalog
- Schema includes: table names, columns, types, descriptions, sample data
- Table Agent generates SQL queries aware of the full schema (Databricks-style)
- Read-only enforcement at query time (SELECT only)
- **Merged-cell repair**: PyMuPDF's `find_tables()` sometimes packs a whole row of period values into a single cell; `_split_merged_numeric_cells` + `_infer_extra_col_names` detect and split them, auto-incrementing year-named columns (`col_2018` → `col_2019` …)
- **Tables-only maintenance**: `python reextract_tables.py` drops the SQLite side and re-scans PDFs without touching FAISS/BM25/RAPTOR; `python inspect_tables.py` prints schemas + samples

---

## State & Memory

### Short-term (per session)
- **LangGraph SqliteSaver checkpointer** — auto-persists full state per `thread_id`
- Conversation history via `add_messages` reducer — auto-accumulated
- Follow-up detection uses message history

### Long-term (cross session)
- **LangGraph InMemoryStore** — namespaced by `user_id`
- Stores past Q&A summaries with topics and entities
- Semantic search for relevant past interactions

---

## Observability

### Local JSONL Tracing
- LangSmith was removed (free-tier quota); replaced by `utils/tracing.py`
- Every agent node still wears the `@traceable(run_type="chain", name="...")` decorator — same surface, different sink
- Writes one event per node to `data/traces/trace_<YYYY-MM-DD>.jsonl` with: `ts, run_id, name, run_type, elapsed_ms, error`
- Controlled by `LOCAL_TRACING` env var (default `true`)

### Execution Trace
- Every agent appends to `execution_trace` in state (reducer: `operator.add`)
- Shows: agent name, action, input/output summary, duration
- Displayed in the UI's Execution Trace panel and in the SSE `event: step` summaries

---

## Hallucination Control (defense-in-depth)

1. **Strict prompting**: "Use ONLY the provided context. If not found, emit the exact refusal string." Compound queries answer-what-you-can / gap-what-you-can't instead of refusing wholesale
2. **Two exact refusal strings**: one for out-of-scope, one for ESG-but-not-in-context — preserved verbatim, only emitted when NO part of the query has supporting data
3. **Citation enforcement**: Answer must reference `[Document, Page X]` for every claim; derived numbers cite the rows/columns they were computed from
4. **Entity-metric extractor** (`agents/entity_metric_extractor.py`): pre-synthesis pass that emits atomic `(entity, metric, value, unit, source_chunk_id)` facts. The Reasoning prompt sources EVERY numeric claim from this table — eliminating row-alignment drift in list-shaped passages
5. **Citation verifier** (`agents/citation_verifier.py`): extractive check — every `[Doc, Page N]` must resolve to a chunk we actually retrieved. Coverage < threshold → forced rewrite
6. **Attribution verifier** (`agents/attribution_verifier.py`): deterministic post-LLM check. For every numeric mention in the answer, finds the bound entity by token-proximity scan and looks up the `(entity, value, unit)` triple in the facts table. Classifies failures as `wrong_entity` / `no_supporting_fact` / `unit_mismatch` / `value_mismatch`. Score < threshold → forced rewrite with targeted `rewrite_hint`. No LLM call
7. **Faithfulness check (merged into validation)** (`agents/validation.py`): claim-level decomposition runs inside the validation LLM call. Decomposes the answer into atomic claims and judges each `yes/partial/no` against context. Score < threshold → forced rewrite. Previously a separate `faithfulness_checker.py` — merged so the answer + context are sent once, not twice
8. **Contradiction detector** (`agents/contradiction_detector.py`): one structured LLM call scans retrieved chunks for numeric / factual conflicts about the same metric, year, or scope (chunks vs. each other, not answer vs. chunks). Surfaced in a dedicated UI callout. Skipped on refusals / single-chunk contexts
9. **Calculator** (`agents/calculator.py`): deterministic post-pass that parses every `| Metric | A | B | Δ absolute | Δ relative |` delta table in the answer, recomputes the last two columns, and overwrites the LLM's values if they're off by more than the tolerance. No LLM call, no new dependencies
10. **Validation Agent (CRAG)**: grounding LLM judge that can pass / rewrite_answer / re_retrieve / give_up — in the same call as the claim-level faithfulness decomposition
11. **CRAG loop**: max 2 retries per path (`validation_retries`); attribution failures generate a targeted `rewrite_hint` that the next reasoning pass consumes
12. **Confidence scoring**: 0.0 (no support) to 1.0 (fully grounded), surfaced as a UI quality pill
13. **Information gaps**: system explicitly reports what was NOT found, both as a structured field and in the answer body
14. **Adversarial JSONL harness**: `evaluation/adversarial_eval.py` runs cases across wrong-year / missing-data / multi-hop / citation-fidelity categories with per-category pass criteria

---

## Explainability (Output)

Every response includes:
- **Answer** — adaptive markdown: lead paragraph + organic section headers + bulleted entity lists + optional Bottom-line callout + required delta tables for YoY comparisons + Information gaps, fully cited
- **Citations** — document name, page number, relevance score
- **Confidence score** — model's own narrative confidence in the answer
- **Citation coverage** — fraction of `[Doc, Page N]` markers that resolved to actually-retrieved chunks
- **Faithfulness score** — fraction of atomic claims judged supported by context (now produced inside the validation call)
- **Attribution score** — fraction of numeric claims that resolved to a fact in the pre-extracted facts table
- **Attribution mismatches** — structured list of `wrong_entity` / `no_supporting_fact` / `unit_mismatch` / `value_mismatch` failures
- **Verified / unverified citations** — split list with reasons for the unverified ones
- **Claim verdicts** — every atomic claim + a yes/partial/no judgment
- **Facts** — the pre-extracted `(entity, metric, value, unit)` table used by the Reasoning agent
- **Contradictions** — structured list of cross-chunk conflicts surfaced by the contradiction detector
- **Arithmetic corrections** — list of `(metric, original, corrected)` triples for any delta-table values rewritten by the calculator
- **Execution trace** — step-by-step agent actions with timing (also streamed live via SSE)
- **Reasoning summary** — how the answer was derived
- **Information gaps** — what's missing from the context
- **Generated SQL** — if table agent was used
- **Query classification** — type, scope, intent, complexity_tier
- **Sub-questions / sub-question evidence / mentioned documents** — visible in the API response when the query was decomposed or had explicit entities

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph StateGraph (1.0+) |
| LLM (light) | Ollama: `gpt-oss:120b-cloud` (or any model you point `OLLAMA_MODEL_LIGHT` at) |
| LLM (heavy) | Ollama: `gpt-oss:120b-cloud` (or any model you point `OLLAMA_MODEL_HEAVY` at) |
| LLM (vision) | Ollama: `llava` (preprocessing only, optional via `--no-vision` / `--no-images`) |
| Embeddings | `mxbai-embed-large` via Ollama (1024-dim, 512-token context) |
| Vector DB | FAISS (`IndexFlatIP` over L2-normalized vectors = cosine similarity) |
| Keyword search | rank_bm25 (rebuilt from FAISS docstore on uvicorn startup) |
| Reranking | sentence-transformers cross-encoder (`ms-marco-MiniLM-L-6-v2`), optional (`rerank_enabled`) |
| PDF processing | PyMuPDF + pymupdf4llm (configurable via `pdf_extractor`) |
| OCR | EasyOCR (pip-installable, no system deps) |
| Table storage | SQLite + pandas (with merged-cell repair) |
| State persistence | LangGraph SqliteSaver (`data/checkpoints.db`) |
| Long-term memory | LangGraph InMemoryStore (per `user_id`) |
| Tracing | Local JSONL (`utils/tracing.py` → `data/traces/`) |
| API | FastAPI + Uvicorn (with SSE streaming on `/query/stream`) |
| UI | HTML + CSS + Vanilla JS (light theme locked, markdown-table rendering, live SSE step list) |
| Evaluation | Unified runner (`evaluation/run_all.py`) over agentic / retrieval / adversarial / gold-builder sub-evaluators; RAGAS removed |

---

## How to Run

The browser UI is the minimum-friction path — no preprocessing CLI needed.

```bash
# 1. Install
pip install -r requirements.txt

# 2. Install Ollama (https://ollama.com/download), then pull models referenced in .env
ollama pull mxbai-embed-large
ollama signin                          # only if using -cloud models like gpt-oss:120b-cloud

# 3. (Optional) override defaults in .env — see config/settings.py for the full list

# 4. Run the app
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** → click **Upload PDF** in the sidebar → drop a file → wait for the live progress (`Checking Ollama → Wiping data → Saving <file>.pdf → Extracting text → Chunking → Tables → Images → Contextualizing → Indexing → BM25 → RAPTOR → Schema catalog → Complete`) → start asking questions.

If Ollama isn't running or a model isn't pulled, the upload modal shows in-app install / pull / signin instructions per-OS and a Retry button — no terminal round-trip needed.

### CLI alternative

```bash
python preprocessing.py --clear              # ingest whatever is in data/pdfs/
python main.py query "What are the main sustainability themes?"
python main.py chat                          # interactive REPL
```

---

## Project Structure

```
agentic_rag/
├── app.py                       # FastAPI entry point (+ /query/stream SSE + startup pre-warm)
├── main.py                      # CLI entry point
├── preprocessing.py             # Single-file PDF pipeline (text + tables + images + RAPTOR)
├── reextract_tables.py          # Drop SQLite tables + re-scan PDFs (no FAISS rebuild)
├── inspect_tables.py            # CLI inspector for data/tables.db
├── wipe_data.py                 # Clear preprocessed data; preserve PDFs + model cache
├── generate_workflow_png.py     # Self-contained LangGraph diagram renderer
├── requirements.txt
├── SUMMARY.md                   # This file
├── README.md
├── CLAUDE.md                    # Guidance for Claude Code working in this repo
├── config/settings.py           # Centralized Pydantic Settings
├── graph/
│   ├── state.py                 # AgentState + Pydantic structured output models
│   ├── workflow.py              # LangGraph StateGraph + routers + smalltalk pre-flight
│   └── checkpointer.py          # SqliteSaver(check_same_thread=False) + InMemoryStore
├── agents/
│   ├── query_understanding.py    # + mentioned_documents extraction
│   ├── planner.py                # End-early / RAG / Table / Both + sub-questions + complexity_tier
│   ├── retrieval.py              # Hybrid + RAPTOR + Map-Reduce + metadata pre-filter + per-source cap + neighbor expansion
│   ├── table_agent.py            # Text-to-SQL
│   ├── entity_metric_extractor.py # Pre-synthesis (entity, metric, value, unit) fact extraction
│   ├── reasoning.py              # Synthesis (adaptive markdown) + post-LLM verifiers
│   ├── citation_verifier.py      # Extractive [Doc, Page N] check
│   ├── attribution_verifier.py   # Deterministic entity↔value attribution check
│   ├── contradiction_detector.py # Cross-chunk numeric / factual conflict scan
│   ├── calculator.py             # Deterministic arithmetic verifier for delta tables
│   ├── validation.py             # CRAG + merged claim-level faithfulness in one LLM call + hard overrides
│   └── memory_agent.py           # LangGraph Store read/write
├── retrieval/
│   ├── chunking.py               # Parent-child chunking
│   ├── vector_store.py           # FAISS (IndexFlatIP cosine) + $any_in / $contains_any
│   ├── bm25_retriever.py         # BM25 (in-memory, rebuilt at uvicorn startup)
│   ├── hybrid.py                 # RRF fusion
│   ├── reranker.py               # Cross-encoder (optional via rerank_enabled)
│   └── raptor.py                 # RAPTOR tree
├── storage/
│   ├── sql_store.py              # SQLite (SELECT-only at runtime)
│   └── schema_manager.py         # Schema catalog (cached)
├── evaluation/                   # Unified eval suite
│   ├── run_all.py                # Entry point: agentic / retrieval / adversarial / gold / all
│   ├── agentic_eval.py           # End-to-end answer quality vs QNA_data/*.json (LLM-judge + cosine)
│   ├── retrieval_eval.py         # Dense vs hybrid vs hybrid+rerank on gold_retrieval.jsonl
│   ├── adversarial_eval.py       # Robustness / refusal / multi-hop / citation fidelity
│   ├── build_gold_dataset.py     # LLM-synthesized retrieval gold set generator
│   ├── response_logger.py        # Shared per-query trace logger
│   ├── adversarial_questions.jsonl
│   └── gold_retrieval.jsonl
├── QNA_data/                     # (untracked) human-labelled Q/A JSON files for agentic eval
├── utils/
│   ├── llm.py                    # LLM routing (light/heavy/vision/embed) + dynamic num_ctx + context budgets
│   ├── embeddings.py             # Ollama embedding wrapper (1200-char cap)
│   ├── tracing.py                # Local JSONL tracer (replaces LangSmith)
│   └── logging_config.py         # Structured logging + UTF-8 stdout on Windows
├── static/                       # Light theme UI (markdown tables + live SSE step list)
│   ├── index.html
│   ├── app.js
│   └── style.css
└── data/                         # PDFs, FAISS, tables.db, traces (all gitignored)
```



