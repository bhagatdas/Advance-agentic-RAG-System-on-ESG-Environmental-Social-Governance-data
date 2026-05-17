# Agentic RAG System for ESG (Environmental, Social, and Governance) Data

> **ESG Insight Pro** — a multi-agent Retrieval-Augmented Generation system that answers questions about ESG / Sustainability PDF reports with grounded, cited responses.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-1c3d5a.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1c3d5a.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-4b3f72.svg)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey.svg)](#license)

Built on **LangGraph**, **Ollama**, and **ChromaDB**. Ingests PDFs (text + tables + images), builds a hierarchical retrieval index, and answers questions through a **7-agent workflow** with corrective feedback loops (CRAG).

---

## Workflow

![LangGraph Workflow](workflow.png)

The planner decides **per query**:
1. **In-scope check** — greetings, smalltalk, or off-topic queries short-circuit straight to `END` with a polite redirect.
2. **Context source** — RAG only, Table only, or **both** (richer mixed context).
3. **RAG strategy** — `standard` / `raptor_global` / `map_reduce`.

---

## Features

- **Multi-modal ingestion** — text + tables (extracted into SQLite) + images (OCR via EasyOCR + vision-LLM captions)
- **Parent-child chunking** with **Anthropic-style contextual retrieval** (chunks prepended with 2-3 sentence context)
- **Hybrid retrieval** — BM25 (sparse) + ChromaDB (dense) merged with Reciprocal Rank Fusion
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) for precision filtering
- **RAPTOR tree** — recursive K-Means clustering + LLM summarization, 3 levels deep, for global queries
- **Map-Reduce** strategy for exhaustive aggregation (`"list ALL initiatives..."`)
- **Schema-aware NL → SQL** for tabular queries, with SELECT-only safety enforcement
- **7 specialized agents** with Pydantic structured output
- **CRAG corrective loop** — validation can trigger re-retrieve / re-reason (≤2 retries each)
- **Memory** — SqliteSaver checkpointer (short-term) + Store (long-term, per-user)
- **Observability** — LangSmith tracing on every node + per-agent execution trace in the API response
- **Three interfaces** — FastAPI REST API, browser chat UI, CLI

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# edit .env -- set LANGCHAIN_API_KEY and confirm Ollama model names

# 3. Start Ollama (separate terminal) and pull the models referenced in .env
ollama serve
ollama pull nomic-embed-text
# ... plus your light/heavy/vision models

# 4. Drop ESG PDFs into data/pdfs/ then ingest
python main.py ingest

# 5a. CLI chat
python main.py chat

# 5b. or REST API + browser UI
uvicorn app:app --reload --port 8000
# -> http://localhost:8000
```

---

## CLI

```bash
python main.py ingest                                 # Preprocess all PDFs
python main.py ingest --clear --no-vision             # Re-ingest, skip vision captions
python main.py query "What are Scope 1 emissions?"    # Single query
python main.py chat                                   # Interactive REPL
python main.py schema                                 # Print schema catalog
```

---

## REST API

| Method | Path       | Purpose                                                             |
|:------:|:-----------|:--------------------------------------------------------------------|
| `GET`  | `/`        | Serve chat UI                                                       |
| `GET`  | `/health`  | Vector store doc count + table count                                |
| `POST` | `/query`   | Run the full pipeline; returns answer + citations + trace + duration |
| `POST` | `/ingest`  | Trigger preprocessing                                               |
| `POST` | `/upload`  | Save uploaded PDF to `data/pdfs/`                                   |
| `GET`  | `/schema`  | Return schema catalog string                                        |

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"How did Scope 1 emissions change year over year and what drove the change?", "user_id":"alice"}'
```

---

## Architecture

```
PDF -> text + tables + images
     -> parent/child chunks -> contextualize -> embed -> ChromaDB
     -> RAPTOR hierarchical summaries
     -> SQLite table store + schema catalog

Query -> Understanding -> Memory Read -> Planner
              |
       +-- end_early (greetings / off-topic) ---------------------> END
       +-- Retrieval (RAG)        -+
       +-- Table Agent (SQL)       +-> Reasoning -> Validation (CRAG) -> Memory Save -> END
       +-- Both (RAG + Table)     -+                    |
                                                        +-- retry retrieval / reasoning (<=2x)
```

Full design, agent-by-agent breakdown, and end-to-end query traces: **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

---

## Tech stack

| Layer | Tech |
|---|---|
| Orchestration | LangGraph 0.4+, LangChain 0.3+ |
| LLM | Ollama (DeepSeek light/heavy, LLaVA vision, Nomic embed) |
| Vector DB | ChromaDB (cosine, HNSW) |
| Sparse retrieval | rank_bm25 |
| Reranker | sentence-transformers cross-encoder |
| Tables | SQLite + pandas |
| PDF | PyMuPDF + pymupdf4llm |
| OCR | EasyOCR (pip-installable) |
| API | FastAPI + Uvicorn |
| Tracing | LangSmith |
| Eval | RAGAS + custom semantic similarity |

---

## Project Structure

```
agentic_rag/
├── main.py                       # CLI entry
├── app.py                        # FastAPI REST API
├── config/settings.py            # Pydantic Settings
├── utils/                        # LLM router, embeddings, cache, logging
├── ingestion/                    # PDF -> text/tables/images -> chunks
│   ├── pdf_processor.py
│   ├── table_extractor.py        # tables -> SQLite
│   ├── image_extractor.py        # OCR + vision captions
│   ├── contextualizer.py         # Anthropic contextual retrieval
│   └── preprocessor.py           # master pipeline
├── retrieval/
│   ├── chunking.py               # parent-child chunks
│   ├── vector_store.py           # ChromaDB
│   ├── bm25_retriever.py
│   ├── hybrid.py                 # RRF fusion
│   ├── reranker.py               # cross-encoder
│   └── raptor.py                 # hierarchical summaries
├── storage/
│   ├── sql_store.py              # SQLite (SELECT-only at runtime)
│   └── schema_manager.py
├── graph/
│   ├── state.py                  # AgentState + Pydantic models
│   ├── checkpointer.py
│   └── workflow.py               # LangGraph StateGraph + routers
├── agents/
│   ├── query_understanding.py
│   ├── planner.py                # decides RAG / Table / Both / End-Early
│   ├── memory_agent.py
│   ├── retrieval.py              # standard / raptor / map-reduce
│   ├── table_agent.py            # NL -> SQL
│   ├── reasoning.py
│   └── validation.py             # CRAG
├── evaluation/                   # RAGAS + custom metrics
├── static/                       # chat UI
└── data/                         # (gitignored) PDFs, ChromaDB, tables.db, logs
```

---

## Configuration

All settings via `.env` (see [.env.example](.env.example)):

| Group       | Keys |
|---|---|
| Ollama      | `OLLAMA_BASE_URL`, `OLLAMA_MODEL_LIGHT`, `OLLAMA_MODEL_HEAVY`, `OLLAMA_MODEL_VISION`, `OLLAMA_MODEL_EMBED` |
| Storage     | `CHROMA_PERSIST_DIR`, `SQLITE_TABLE_DB`, `CHECKPOINT_DB`, `DATA_DIR`, `PDF_DIR`, `IMAGE_DIR` |
| Chunking    | `CHUNK_SIZE_CHILD=400`, `CHUNK_SIZE_PARENT=1600`, `CHUNK_OVERLAP=50` |
| Retrieval   | `RETRIEVAL_TOP_K=20`, `RERANK_TOP_K=5` |
| RAPTOR      | `RAPTOR_CLUSTER_SIZE=10`, `RAPTOR_MAX_LEVELS=3` |
| Memory      | `SHORT_TERM_MAX_MESSAGES=20`, `LONG_TERM_MAX_ITEMS=200` |
| LangSmith   | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` |

---

## Example Query Flows

**Aggregation:** `"List all ESG initiatives mentioned in the Honeywell report"`
1. Understanding → `scope=aggregation`, `entities=["ESG initiatives", "Honeywell"]`
2. Planner → `strategy=map_reduce`, `use_rag=True`, `use_table=False`
3. Retrieval → per-chunk LLM extraction across the corpus, dedup
4. Reasoning → enumerated answer with `[doc, page]` citations
5. Validation → `pass` → memory save → END

**Mixed (number + narrative):** `"How did Scope 1 emissions change YoY and what drove the change?"`
1. Planner → `use_rag=True` AND `use_table=True`
2. Retrieval → standard hybrid search for narrative context
3. Table Agent → generates SQL on extracted emissions table
4. Reasoning → fuses both sources, cites narrative chunk + table row
5. Validation → `pass` → END

**Smalltalk:** `"hi"` → Planner short-circuits to `END` with a polite redirect. No retrieval, no LLM reasoning cost.

---

## Documentation

- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** — full architecture & agent details
- **[workflow.png](workflow.png)** — rendered LangGraph state diagram
- **[workflow.mmd](workflow.mmd)** — editable Mermaid source

---

## License

Proprietary — see repository owner.

---

*Built by [Bhagat Labs](https://github.com/bhagatdas).*
