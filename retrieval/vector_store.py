"""
FAISS-backed vector store with the same API surface as the previous ChromaDB
implementation, so callers (agents/retrieval.py, retrieval/hybrid.py,
preprocessing.py, app.py) need no changes.

Layout on disk (under settings.faiss_persist_dir):
    index.faiss    — FAISS IndexFlatIP over L2-normalized vectors (== cosine similarity)
    docstore.json  — sidecar with the doc records and chunk_id <-> position map

Metadata filtering supports the subset of Mongo-style operators actually used
in this codebase:
    {"field": value}                 -> equality
    {"field": {"$ne": v}}            -> not equal
    {"field": {"$gte": v}}           -> >=
    {"field": {"$lte": v}}           -> <=
    {"field": {"$eq": v}}            -> ==
    {"$and": [clause, clause, ...]}  -> conjunction
Multiple top-level keys are an implicit AND.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.settings import settings
from utils.embeddings import embed_texts, embed_text

logger = logging.getLogger(__name__)


# ── Tiny Mongo-style filter matcher ──────────────────────────────────────────

def _match_clause(metadata: dict, field: str, condition) -> bool:
    """Match one (field, condition) pair."""
    value = metadata.get(field)
    if isinstance(condition, dict):
        for op, target in condition.items():
            if op == "$eq" and value != target:
                return False
            elif op == "$ne" and value == target:
                return False
            elif op == "$gte" and (value is None or value < target):
                return False
            elif op == "$lte" and (value is None or value > target):
                return False
            elif op == "$gt" and (value is None or value <= target):
                return False
            elif op == "$lt" and (value is None or value >= target):
                return False
            elif op == "$in" and value not in target:
                return False
            elif op == "$nin" and value in target:
                return False
        return True
    return value == condition


def _matches(metadata: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    for key, condition in where.items():
        if key == "$and":
            if not all(_matches(metadata, sub) for sub in condition):
                return False
        elif key == "$or":
            if not any(_matches(metadata, sub) for sub in condition):
                return False
        else:
            if not _match_clause(metadata, key, condition):
                return False
    return True


# ── Vector store ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS IndexFlatIP over L2-normalized embeddings (cosine similarity).
    Persists the index + a JSON sidecar holding doc content/metadata.
    """

    def __init__(
        self,
        collection_name: str = "sustainability_chunks",
        persist_dir: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir or settings.faiss_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self.persist_dir / f"{collection_name}.faiss"
        self._docstore_path = self.persist_dir / f"{collection_name}.json"
        self._lock = threading.Lock()

        # Probe dimension from a tiny embedding
        self._dim = self._probe_dim()

        self._index: faiss.Index
        self._docs: list[dict] = []          # position -> {"id", "content", "metadata"}
        self._id_to_pos: dict[str, int] = {} # chunk_id -> position

        self._load_or_init()

        logger.info(
            "Vector store ready — collection=%s, dim=%d, docs=%d, dir=%s",
            collection_name, self._dim, len(self._docs), self.persist_dir,
        )

    # ── Persistence ───────────────────────────────────────────────────────

    def _probe_dim(self) -> int:
        vec = embed_text("dimension probe")
        return len(vec)

    def _new_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(self._dim)

    def _load_or_init(self) -> None:
        if self._index_path.exists() and self._docstore_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                with open(self._docstore_path, "r", encoding="utf-8") as f:
                    docstore = json.load(f)
                self._docs = docstore.get("docs", [])
                self._id_to_pos = docstore.get("id_to_pos", {})
                if self._index.ntotal != len(self._docs):
                    logger.warning(
                        "FAISS index / docstore size mismatch (%d vs %d) — rebuilding empty",
                        self._index.ntotal, len(self._docs),
                    )
                    self._index = self._new_index()
                    self._docs = []
                    self._id_to_pos = {}
                return
            except Exception as e:
                logger.warning("Failed to load existing FAISS state: %s — starting fresh", e)

        self._index = self._new_index()
        self._docs = []
        self._id_to_pos = {}

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        with open(self._docstore_path, "w", encoding="utf-8") as f:
            json.dump({"docs": self._docs, "id_to_pos": self._id_to_pos}, f)

    # ── Write ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add chunks to the index. Each chunk dict needs:
          content: str
          chunk_id: str
          document_name, page_number, chunk_type, parent_id, raptor_level (optional)
        Chunks whose chunk_id is already present are skipped (FAISS flat indexes
        don't support cheap in-place updates — use clear() to rebuild).
        """
        if not chunks:
            return

        new_records: list[dict] = []
        new_texts: list[str] = []
        skipped = 0

        with self._lock:
            for chunk in chunks:
                chunk_id = chunk.get("chunk_id") or f"chunk_{len(self._docs) + len(new_records)}"
                if chunk_id in self._id_to_pos:
                    skipped += 1
                    continue
                record = {
                    "id": chunk_id,
                    "content": chunk.get("content", ""),
                    "metadata": {
                        "document_name": chunk.get("document_name", ""),
                        "page_number": chunk.get("page_number", 0),
                        "chunk_type": chunk.get("chunk_type", "child"),
                        "parent_id": chunk.get("parent_id", ""),
                        "raptor_level": chunk.get("raptor_level", 0),
                    },
                }
                new_records.append(record)
                new_texts.append(record["content"])

            if not new_records:
                if skipped:
                    logger.info("add_chunks: all %d chunks already indexed", skipped)
                return

            # Embed + index + persist in batches so a mid-flight failure
            # doesn't lose all earlier progress.
            batch = 64
            embedded_count = 0
            for i in range(0, len(new_texts), batch):
                batch_texts = new_texts[i:i + batch]
                batch_records = new_records[i:i + batch]
                try:
                    vecs = embed_texts(batch_texts)
                except Exception as e:
                    logger.error(
                        "Embedding failed at batch starting %d (size %d): %s — skipping batch",
                        i, len(batch_texts), e,
                    )
                    continue

                arr = np.asarray(vecs, dtype=np.float32)
                faiss.normalize_L2(arr)
                self._index.add(arr)

                start = len(self._docs)
                for offset, record in enumerate(batch_records):
                    self._docs.append(record)
                    self._id_to_pos[record["id"]] = start + offset

                embedded_count += len(batch_records)
                self._persist()  # checkpoint after every batch

        logger.info(
            "Indexed %d new chunks (skipped %d already-present, %d batch failures) — total=%d",
            embedded_count, skipped, len(new_records) - embedded_count, len(self._docs),
        )

    # ── Read ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,  # accepted for API parity, ignored
    ) -> list[dict]:
        if self._index.ntotal == 0:
            return []

        # Over-fetch when filtering since we apply the predicate post-hoc
        fetch_k = top_k * 5 if where else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        qvec = np.asarray([embed_text(query)], dtype=np.float32)
        faiss.normalize_L2(qvec)
        scores, indices = self._index.search(qvec, fetch_k)

        results = []
        for pos, score in zip(indices[0].tolist(), scores[0].tolist()):
            if pos < 0 or pos >= len(self._docs):
                continue
            doc = self._docs[pos]
            if where and not _matches(doc["metadata"], where):
                continue
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score),
            })
            if len(results) >= top_k:
                break

        logger.debug("Vector search — query='%s...', results=%d", query[:50], len(results))
        return results

    def search_by_raptor_level(
        self,
        query: str,
        min_level: int = 0,
        max_level: int = 3,
        top_k: int = 10,
    ) -> list[dict]:
        return self.search(
            query=query,
            top_k=top_k,
            where={"$and": [
                {"raptor_level": {"$gte": min_level}},
                {"raptor_level": {"$lte": max_level}},
            ]},
        )

    def get_parent_chunk(self, parent_id: str) -> Optional[dict]:
        for doc in self._docs:
            meta = doc["metadata"]
            if meta.get("parent_id") == parent_id and meta.get("chunk_type") == "parent":
                return {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": meta,
                }
        return None

    def get_all_documents(self) -> list[dict]:
        """Return all non-parent docs (for BM25 indexing)."""
        return [
            {"id": d["id"], "content": d["content"], "metadata": d["metadata"]}
            for d in self._docs
            if d["metadata"].get("chunk_type") != "parent"
        ]

    # ── Admin ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        with self._lock:
            self._index = self._new_index()
            self._docs = []
            self._id_to_pos = {}
            # Remove on-disk artifacts
            for p in (self._index_path, self._docstore_path):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
        logger.info("Vector store cleared")

    @property
    def count(self) -> int:
        return len(self._docs)


# Singleton
vector_store = VectorStore()
