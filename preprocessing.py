"""
PDF → FAISS preprocessing pipeline (single-file).

Replaces the previous ingestion/ package. Does the entire offline ingestion flow:

  PDF
    -> text extraction (PyMuPDF)
    -> table extraction (PyMuPDF find_tables -> pandas -> SQLite)
    -> image extraction (EasyOCR + LLaVA captions)
    -> parent/child chunking
    -> Anthropic-style contextualization
    -> embedding (Ollama)
    -> FAISS index upsert
  + BM25 index build over the corpus
  + RAPTOR hierarchical summary tree (KMeans + LLM summarize)
  + Schema catalog generation

Run as CLI:
    python preprocessing.py
    python preprocessing.py --clear --no-vision
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config.settings import settings
from retrieval.vector_store import vector_store
from retrieval.bm25_retriever import bm25_index
from storage.sql_store import sql_store
from storage.schema_manager import schema_manager
from utils.embeddings import embed_texts
from utils.llm import invoke_llm, invoke_vision_llm
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Data classes
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PageText:
    page_number: int
    text: str
    document_name: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class ExtractedTable:
    page_number: int
    table_index: int
    document_name: str
    dataframe: pd.DataFrame
    sql_table_name: str = ""
    text_representation: str = ""
    description: str = ""


@dataclass
class ExtractedImage:
    page_number: int
    image_index: int
    document_name: str
    image_path: str
    ocr_text: str = ""
    caption: str = ""
    combined_text: str = ""

    def __post_init__(self):
        parts = []
        if self.caption:
            parts.append(f"Image description: {self.caption}")
        if self.ocr_text:
            parts.append(f"Text in image: {self.ocr_text}")
        if not parts:
            parts.append("(Image with no extractable text content)")
        self.combined_text = (
            f"[IMAGE from {self.document_name}, Page {self.page_number}]\n"
            + "\n".join(parts)
        )


# ════════════════════════════════════════════════════════════════════════════
# 1. PDF text extraction
# ════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(
    pdf_path: str,
    ocr_fallback: bool = True,
    ocr_min_chars: int = 100,
    ocr_dpi: int = 200,
) -> list[PageText]:
    """
    Extract clean text per page. When PyMuPDF returns < ocr_min_chars for a
    page (typical for scanned / image-only pages), render the page and run
    EasyOCR as a fallback; the OCR text is used iff it found more content.
    OCR text is saved in the same `text` field as PyMuPDF text — downstream
    chunking / indexing treats it identically.
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    doc_name = path.stem
    pages: list[PageText] = []
    ocr_pages_used = 0

    try:
        doc = fitz.open(str(path))
        logger.info(
            "Processing PDF — file=%s, pages=%d, ocr_fallback=%s",
            doc_name, len(doc), ocr_fallback,
        )
        for page_num in range(len(doc)):
            page = doc[page_num]
            cleaned = _clean_text(page.get_text("text"))

            if ocr_fallback and len(cleaned) < ocr_min_chars:
                ocr_text = _ocr_page(page, dpi=ocr_dpi)
                if len(ocr_text) > len(cleaned):
                    logger.info(
                        "Page %d OCR fallback — pymupdf=%d chars, ocr=%d chars",
                        page_num + 1, len(cleaned), len(ocr_text),
                    )
                    cleaned = _clean_text(ocr_text)
                    ocr_pages_used += 1

            if cleaned.strip():
                pages.append(PageText(
                    page_number=page_num + 1,
                    text=cleaned,
                    document_name=doc_name,
                ))
        doc.close()
        total_chars = sum(p.char_count for p in pages)
        logger.info(
            "Text extraction complete — doc=%s, pages=%d, chars=%d, ocr_pages=%d",
            doc_name, len(pages), total_chars, ocr_pages_used,
        )
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", pdf_path, e)

    return pages


def _ocr_page(page, dpi: int = 200) -> str:
    """Render a PDF page to a PNG and OCR it via EasyOCR. Returns merged text."""
    try:
        reader = _get_ocr_reader()
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        results = reader.readtext(png_bytes, detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        logger.warning("Page OCR failed: %s", e)
        return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "" or len(stripped) > 3:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


# ════════════════════════════════════════════════════════════════════════════
# 2. Table extraction (-> SQLite + text representation for vector search)
# ════════════════════════════════════════════════════════════════════════════

def extract_tables_from_pdf(pdf_path: str) -> list[ExtractedTable]:
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    doc_name = path.stem
    extracted: list[ExtractedTable] = []

    try:
        doc = fitz.open(str(path))
        logger.info("Scanning for tables — file=%s, pages=%d", doc_name, len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                tables = page.find_tables()
            except Exception:
                continue

            for table_idx, table in enumerate(tables):
                try:
                    df = table.to_pandas()
                    if df.empty or df.shape[0] < 2:
                        continue
                    if _looks_like_header(df):
                        df.columns = df.iloc[0].astype(str)
                        df = df.iloc[1:].reset_index(drop=True)
                    df = _clean_dataframe(df)
                    if df.empty:
                        continue

                    table_name = f"{doc_name}_p{page_num + 1}_t{table_idx + 1}"
                    description = _generate_table_description(df, doc_name, page_num + 1)
                    actual_name = sql_store.create_table(
                        table_name=table_name,
                        df=df,
                        source_document=path.name,
                        source_page=page_num + 1,
                        description=description,
                    )
                    text_repr = _create_text_representation(df, description, doc_name, page_num + 1)
                    extracted.append(ExtractedTable(
                        page_number=page_num + 1,
                        table_index=table_idx,
                        document_name=doc_name,
                        dataframe=df,
                        sql_table_name=actual_name,
                        text_representation=text_repr,
                        description=description,
                    ))
                    logger.info(
                        "Table extracted — page=%d, idx=%d, shape=%s, sql=%s",
                        page_num + 1, table_idx, df.shape, actual_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed table on page %d, idx %d: %s", page_num + 1, table_idx, e
                    )

        doc.close()
        logger.info("Table extraction complete — doc=%s, tables=%d", doc_name, len(extracted))
    except Exception as e:
        logger.error("Failed to extract tables from %s: %s", pdf_path, e)

    return extracted


def _looks_like_header(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    first_row = df.iloc[0]
    str_count = sum(1 for v in first_row if isinstance(v, str) and not _is_numeric(str(v)))
    return str_count > len(first_row) * 0.5


def _is_numeric(s: str) -> bool:
    try:
        float(s.replace(",", "").replace("%", "").replace("$", ""))
        return True
    except (ValueError, AttributeError):
        return False


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    for col in df.columns:
        try:
            numeric_col = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("%", "").str.replace("$", ""),
                errors="coerce",
            )
            if numeric_col.notna().sum() > len(df) * 0.5:
                df[col] = numeric_col
        except Exception:
            pass
    return df


def _generate_table_description(df: pd.DataFrame, doc_name: str, page: int) -> str:
    try:
        preview = df.head(3).to_string(index=False)
        columns = ", ".join(df.columns.tolist())
        prompt = (
            f"Describe this table from '{doc_name}' (page {page}) in one sentence.\n"
            f"Focus on what data it contains and its purpose.\n\n"
            f"Columns: {columns}\nSample data:\n{preview}\n\n"
            "Description (one sentence):"
        )
        return invoke_llm(prompt, task_type="light").strip()[:500]
    except Exception as e:
        logger.warning("Failed to generate table description: %s", e)
        return f"Table from {doc_name}, page {page} with columns: {', '.join(df.columns.tolist()[:10])}"


def _create_text_representation(
    df: pd.DataFrame, description: str, doc_name: str, page: int
) -> str:
    return "\n".join([
        f"[TABLE from {doc_name}, Page {page}]",
        f"Description: {description}",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Number of rows: {len(df)}",
        "",
        "Data preview:",
        df.head(5).to_string(index=False),
    ])


# ════════════════════════════════════════════════════════════════════════════
# 3. OCR (EasyOCR, lazy-loaded)
# ════════════════════════════════════════════════════════════════════════════

_ocr_reader = None


def _get_ocr_reader(languages: Optional[list[str]] = None):
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        langs = languages or ["en"]
        logger.info("Initializing EasyOCR reader — languages=%s", langs)
        _ocr_reader = easyocr.Reader(langs, gpu=False)
    return _ocr_reader


def ocr_image(image_path: str, languages: Optional[list[str]] = None) -> str:
    path = Path(image_path)
    if not path.exists():
        return ""
    try:
        reader = _get_ocr_reader(languages)
        results = reader.readtext(str(path), detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        logger.error("OCR failed for %s: %s", image_path, e)
        return ""


# ════════════════════════════════════════════════════════════════════════════
# 4. Image extraction (+ caption via vision LLM)
# ════════════════════════════════════════════════════════════════════════════

def extract_images_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    use_vision_model: bool = True,
    min_image_size: int = 100,
) -> list[ExtractedImage]:
    path = Path(pdf_path)
    if not path.exists():
        return []

    doc_name = path.stem
    img_dir = Path(output_dir or settings.image_dir) / doc_name
    img_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[ExtractedImage] = []

    try:
        doc = fitz.open(str(path))
        logger.info("Scanning for images — file=%s, pages=%d", doc_name, len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue

                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if width < min_image_size or height < min_image_size:
                        continue

                    ext = base_image.get("ext", "png")
                    img_filename = f"{doc_name}_p{page_num + 1}_img{img_idx + 1}.{ext}"
                    img_path = img_dir / img_filename
                    with open(img_path, "wb") as f:
                        f.write(base_image["image"])

                    ocr_text = ocr_image(str(img_path))
                    caption = ""
                    if use_vision_model:
                        caption = _generate_caption(str(img_path), doc_name, page_num + 1)

                    extracted.append(ExtractedImage(
                        page_number=page_num + 1,
                        image_index=img_idx,
                        document_name=doc_name,
                        image_path=str(img_path),
                        ocr_text=ocr_text,
                        caption=caption,
                    ))
                except Exception as e:
                    logger.warning(
                        "Failed image on page %d idx %d: %s", page_num + 1, img_idx, e
                    )

        doc.close()
        logger.info("Image extraction complete — doc=%s, images=%d", doc_name, len(extracted))
    except Exception as e:
        logger.error("Failed to extract images from %s: %s", pdf_path, e)

    return extracted


def _generate_caption(image_path: str, doc_name: str, page: int) -> str:
    try:
        prompt = (
            f"This image is from '{doc_name}' (page {page}), an ESG/sustainability report. "
            "Describe what this image shows in detail. If it's a chart or graph, "
            "describe the data it presents, including any trends, values, and labels. "
            "If it's a diagram, describe its structure and meaning. "
            "Be specific and factual — this description will be used for information retrieval."
        )
        return invoke_vision_llm(prompt, image_paths=[image_path]).strip()[:1000]
    except Exception as e:
        logger.warning("Failed caption for %s: %s", image_path, e)
        return ""


# ════════════════════════════════════════════════════════════════════════════
# 5. Parent-child chunking
# ════════════════════════════════════════════════════════════════════════════

def create_parent_child_chunks(
    text: str,
    document_name: str,
    page_number: int,
) -> list[dict]:
    child_size = settings.chunk_size_child
    parent_size = settings.chunk_size_parent
    overlap = settings.chunk_overlap

    if not text.strip():
        return []

    chunks = []
    for p_idx, parent_text in enumerate(_split_text(text, parent_size, overlap=overlap * 2)):
        parent_id = f"{document_name}_p{page_number}_parent{p_idx}"
        chunks.append({
            "content": parent_text,
            "document_name": document_name,
            "page_number": page_number,
            "chunk_type": "parent",
            "parent_id": parent_id,
            "chunk_id": parent_id,
            "raptor_level": -1,
        })
        for c_idx, child_text in enumerate(_split_text(parent_text, child_size, overlap=overlap)):
            child_id = f"{document_name}_p{page_number}_parent{p_idx}_child{c_idx}"
            chunks.append({
                "content": child_text,
                "document_name": document_name,
                "page_number": page_number,
                "chunk_type": "child",
                "parent_id": parent_id,
                "chunk_id": child_id,
                "raptor_level": 0,
            })
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sentences = [s for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    chunks: list[str] = []
    current = ""
    current_sents: list[str] = []
    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent
            current_sents.append(sent)
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = ""
            for s in reversed(current_sents):
                if len(overlap_text) + len(s) <= overlap:
                    overlap_text = s + overlap_text
                else:
                    break
            current = overlap_text + sent
            current_sents = [sent]
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# 6. Anthropic-style contextual retrieval
# ════════════════════════════════════════════════════════════════════════════

CONTEXTUALIZE_PROMPT = """You are given a chunk from a document. Provide a short (2-3 sentence) context
that situates this chunk within the broader document. Include: what document this is from,
what section/topic is being discussed, and any critical context needed to understand this chunk in isolation.

Document: {doc_title}
Page: {page_number}

Surrounding text (before this chunk):
{preceding_text}

CHUNK TO CONTEXTUALIZE:
{chunk_text}

CONTEXT (2-3 sentences, be concise):"""


def contextualize_chunks_batch(chunks: list[dict], doc_title: str) -> list[dict]:
    logger.info("Contextualizing %d chunks for '%s'", len(chunks), doc_title)
    out = []
    for i, chunk in enumerate(chunks):
        preceding = chunks[i - 1]["content"][-300:] if i > 0 else ""
        try:
            prompt = CONTEXTUALIZE_PROMPT.format(
                doc_title=doc_title,
                page_number=chunk.get("page_number", 0),
                preceding_text=preceding or "(start of document)",
                chunk_text=chunk["content"][:800],
            )
            context = invoke_llm(prompt, task_type="light").strip()
            new_content = f"[Context: {context}]\n{chunk['content']}" if context else chunk["content"]
        except Exception as e:
            logger.warning("Contextualize failed (chunk %d): %s", i, e)
            new_content = chunk["content"]
        out.append({**chunk, "content": new_content})
        if (i + 1) % 20 == 0:
            logger.info("Contextualized %d/%d chunks", i + 1, len(chunks))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 7. RAPTOR hierarchical summary tree
# ════════════════════════════════════════════════════════════════════════════

RAPTOR_SUMMARY_PROMPT = """You are summarizing a cluster of related text chunks from an ESG/sustainability report.
Create a comprehensive summary that captures ALL key facts, data points, and themes from these chunks.
The summary should be self-contained — someone reading only this summary should understand the main points.

TEXT CHUNKS:
{chunks_text}

COMPREHENSIVE SUMMARY:"""


def build_raptor_tree(chunks: list[dict]) -> list[dict]:
    """
    Build RAPTOR L1..L3 summary nodes and upsert them into the vector store.
    Returns the list of created summary nodes.
    """
    if not chunks or len(chunks) < 3:
        logger.warning("Too few chunks for RAPTOR tree: %d", len(chunks))
        return []

    cluster_size = settings.raptor_cluster_size
    max_levels = settings.raptor_max_levels

    all_nodes: list[dict] = []
    current_texts = [c["content"] for c in chunks]
    current_ids = [c.get("id", f"chunk_{i}") for i, c in enumerate(chunks)]

    for level in range(1, max_levels + 1):
        if len(current_texts) <= 2:
            logger.info("RAPTOR stopping at level %d (%d nodes)", level, len(current_texts))
            break

        logger.info(
            "RAPTOR L%d — clustering %d nodes (cluster_size=%d)",
            level, len(current_texts), cluster_size,
        )
        clusters = _cluster_texts(current_texts, current_ids, cluster_size)

        next_texts, next_ids = [], []
        for c_idx, cluster in enumerate(clusters):
            summary = _summarize_cluster(cluster["texts"])
            if not summary.strip():
                continue
            node_id = f"raptor_L{level}_C{c_idx}"
            node = {
                "content": summary,
                "chunk_id": node_id,
                "document_name": "raptor_summary",
                "page_number": 0,
                "chunk_type": f"raptor_L{level}",
                "parent_id": node_id,
                "raptor_level": level,
            }
            vector_store.add_chunks([node])
            all_nodes.append(node)
            next_texts.append(summary)
            next_ids.append(node_id)

        current_texts, current_ids = next_texts, next_ids
        logger.info("RAPTOR L%d complete — %d summary nodes", level, len(next_texts))

    logger.info("RAPTOR tree complete — total summary nodes: %d", len(all_nodes))
    return all_nodes


def _cluster_texts(texts: list[str], ids: list[str], cluster_size: int) -> list[dict]:
    if len(texts) <= cluster_size:
        return [{"texts": texts, "ids": ids}]

    embeddings = np.array(embed_texts(texts))
    n_clusters = max(2, min(len(texts) // cluster_size, len(texts) - 1))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    grouped: dict[int, dict] = {}
    for i, label in enumerate(labels):
        grouped.setdefault(int(label), {"texts": [], "ids": []})
        grouped[int(label)]["texts"].append(texts[i])
        grouped[int(label)]["ids"].append(ids[i])
    return list(grouped.values())


def _summarize_cluster(texts: list[str]) -> str:
    combined = "\n---\n".join(texts)
    if len(combined) > 4000:
        combined = combined[:4000] + "\n... (truncated)"
    try:
        return invoke_llm(
            RAPTOR_SUMMARY_PROMPT.format(chunks_text=combined),
            task_type="light",
        ).strip()
    except Exception as e:
        logger.error("RAPTOR summarization failed: %s", e)
        return ""


# ════════════════════════════════════════════════════════════════════════════
# 8. Schema catalog (Databricks-style description for the Table Agent)
# ════════════════════════════════════════════════════════════════════════════

def generate_enhanced_catalog(output_path: str = "./data/schema_catalog.json") -> str:
    schemas = sql_store.get_all_schemas()
    if not schemas:
        logger.warning("No tables found — schema catalog is empty")
        return "NO TABLES AVAILABLE"

    enhanced = []
    for schema in schemas:
        schema["columns"] = _enhance_column_descriptions(schema)
        enhanced.append(schema)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, indent=2, default=str)

    logger.info("Schema catalog saved — tables=%d, path=%s", len(enhanced), output_path)
    schema_manager.invalidate_cache()
    return schema_manager.generate_catalog(force_refresh=True)


def _enhance_column_descriptions(schema: dict) -> list[dict]:
    try:
        col_names = [c["name"] for c in schema["columns"]]
        sample = schema.get("sample_data", [])
        prompt = (
            "Given this database table, provide a brief description for each column.\n"
            f"Table: {schema['table_name']}\n"
            f"Table description: {schema.get('description', 'N/A')}\n"
            f"Source: {schema.get('source_document', 'N/A')} (Page {schema.get('source_page', 'N/A')})\n"
            f"Columns: {', '.join(col_names)}\n"
            f"Sample data: {json.dumps(sample[:2], default=str)}\n\n"
            "For each column, provide a description in this format:\n"
            "column_name: description\n\n"
            "Descriptions:"
        )
        response = invoke_llm(prompt, task_type="light")

        desc_map = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                col, desc = line.split(":", 1)
                desc_map[col.strip().lower().replace(" ", "_")] = desc.strip()

        enhanced = []
        for col in schema["columns"]:
            col_copy = dict(col)
            col_copy["description"] = desc_map.get(col["name"], "")
            enhanced.append(col_copy)
        return enhanced
    except Exception as e:
        logger.warning("Failed to enhance column descriptions: %s", e)
        return schema["columns"]


# ════════════════════════════════════════════════════════════════════════════
# 9. Master pipeline
# ════════════════════════════════════════════════════════════════════════════

def preprocess_all_pdfs(
    pdf_dir: Optional[str] = None,
    use_vision: bool = True,
    use_contextual: bool = True,
    use_raptor: bool = True,
    ocr_fallback: bool = True,
    clear_existing: bool = False,
) -> dict:
    """
    Run the full preprocessing pipeline on every PDF in pdf_dir.

    Args:
        pdf_dir: Directory containing PDF files (default: settings.pdf_dir)
        use_vision: Caption images with the vision LLM
        use_contextual: Apply Anthropic-style contextual prefixes
        use_raptor: Build the RAPTOR hierarchical summary tree
        ocr_fallback: When PyMuPDF returns very little text for a page, render
            the page and OCR it via EasyOCR. The OCR output replaces the page
            text and flows through the normal chunking + indexing path.
        clear_existing: Wipe FAISS + SQLite + BM25 before re-ingesting
    """
    pdf_dir = Path(pdf_dir or settings.pdf_dir)

    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("PDF directory created (empty): %s", pdf_dir)
        return {"status": "no_pdfs", "message": f"Place PDF files in {pdf_dir}"}

    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    if not pdf_files:
        return {"status": "no_pdfs", "message": f"No PDFs found in {pdf_dir}"}

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE STARTED")
    logger.info("PDFs found: %d", len(pdf_files))
    logger.info("Options: vision=%s, contextual=%s, raptor=%s",
                use_vision, use_contextual, use_raptor)
    logger.info("=" * 60)

    start = time.time()

    if clear_existing:
        logger.info("Clearing existing data (FAISS + SQLite tables + BM25)...")
        vector_store.clear()
        sql_store.drop_all_tables()
        bm25_index.clear()

    all_chunks_for_raptor: list[dict] = []
    stats = {
        "pdfs_processed": 0,
        "total_pages": 0,
        "text_chunks": 0,
        "tables_extracted": 0,
        "images_processed": 0,
        "raptor_nodes": 0,
    }

    for pdf_path in pdf_files:
        logger.info("-" * 40)
        logger.info("Processing: %s", pdf_path.name)
        doc_stats = _process_single_pdf(
            str(pdf_path),
            use_vision=use_vision,
            use_contextual=use_contextual,
            ocr_fallback=ocr_fallback,
        )
        stats["pdfs_processed"] += 1
        stats["total_pages"] += doc_stats.get("pages", 0)
        stats["text_chunks"] += doc_stats.get("text_chunks", 0)
        stats["tables_extracted"] += doc_stats.get("tables", 0)
        stats["images_processed"] += doc_stats.get("images", 0)
        all_chunks_for_raptor.extend(doc_stats.get("raw_chunks", []))

    # ── BM25 over the corpus (children + table reprs + image captions) ──
    logger.info("Building BM25 index...")
    all_stored = vector_store.get_all_documents()
    if all_stored:
        bm25_index.build_index(
            corpus=[d["content"] for d in all_stored],
            doc_ids=[d["id"] for d in all_stored],
            metadatas=[d.get("metadata", {}) for d in all_stored],
        )

    # ── RAPTOR ──
    if use_raptor and all_chunks_for_raptor:
        logger.info("Building RAPTOR hierarchical summary tree...")
        nodes = build_raptor_tree(all_chunks_for_raptor)
        stats["raptor_nodes"] = len(nodes)

    # ── Schema catalog ──
    logger.info("Generating schema catalog...")
    generate_enhanced_catalog()

    stats["total_time_seconds"] = round(time.time() - start, 1)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)
    return stats


def _process_single_pdf(
    pdf_path: str,
    use_vision: bool = True,
    use_contextual: bool = True,
    ocr_fallback: bool = True,
) -> dict:
    doc_name = Path(pdf_path).stem
    stats = {"pages": 0, "text_chunks": 0, "tables": 0, "images": 0, "raw_chunks": []}

    # 1. Text (with OCR fallback for image-only pages)
    logger.info("[1/5] Extracting text (OCR fallback=%s)...", ocr_fallback)
    pages = extract_text_from_pdf(pdf_path, ocr_fallback=ocr_fallback)
    stats["pages"] = len(pages)

    # 2. Chunk
    logger.info("[2/5] Chunking (parent-child)...")
    chunks: list[dict] = []
    for page in pages:
        chunks.extend(create_parent_child_chunks(
            text=page.text,
            document_name=page.document_name,
            page_number=page.page_number,
        ))

    # 3. Tables
    logger.info("[3/5] Extracting tables...")
    tables = extract_tables_from_pdf(pdf_path)
    stats["tables"] = len(tables)
    for table in tables:
        if table.text_representation:
            chunks.append({
                "content": table.text_representation,
                "document_name": doc_name,
                "page_number": table.page_number,
                "chunk_type": "table_repr",
                "parent_id": f"table_{table.sql_table_name}",
                "chunk_id": f"table_{table.sql_table_name}_repr",
            })

    # 4. Images
    logger.info("[4/5] Extracting + OCR + captioning images...")
    images = extract_images_from_pdf(pdf_path, use_vision_model=use_vision)
    stats["images"] = len(images)
    for img in images:
        if img.combined_text:
            chunks.append({
                "content": img.combined_text,
                "document_name": doc_name,
                "page_number": img.page_number,
                "chunk_type": "image_caption",
                "parent_id": f"image_{img.page_number}_{img.image_index}",
                "chunk_id": f"img_{doc_name}_p{img.page_number}_i{img.image_index}",
            })

    # Snapshot for RAPTOR (raw, pre-context)
    stats["raw_chunks"] = [
        {"content": c["content"], "id": c.get("chunk_id", ""), "metadata": c}
        for c in chunks
        if c.get("chunk_type") != "table_repr"
    ]

    # 5. Contextualize
    if use_contextual and chunks:
        logger.info("[5/5] Contextualizing chunks...")
        chunks = contextualize_chunks_batch(chunks, doc_title=doc_name)
    else:
        logger.info("[5/5] Skipping contextualization")

    # 6. Index in FAISS
    if chunks:
        logger.info("Indexing %d chunks in FAISS...", len(chunks))
        vector_store.add_chunks(chunks)
        stats["text_chunks"] = len(chunks)

    return stats


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="PDF -> FAISS preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preprocessing.py\n"
            "  python preprocessing.py --clear\n"
            "  python preprocessing.py --no-vision --no-raptor\n"
        ),
    )
    parser.add_argument("--pdf-dir", default=None, help="Directory containing PDFs (default: settings.pdf_dir)")
    parser.add_argument("--no-vision", action="store_true", help="Skip LLaVA image captions")
    parser.add_argument("--no-contextual", action="store_true", help="Skip Anthropic contextualization")
    parser.add_argument("--no-raptor", action="store_true", help="Skip RAPTOR tree build")
    parser.add_argument("--no-ocr-fallback", action="store_true",
                        help="Disable EasyOCR fallback for pages where PyMuPDF returns little text")
    parser.add_argument("--clear", action="store_true", help="Wipe FAISS + tables + BM25 first")
    args = parser.parse_args()

    setup_logging(level="INFO")

    stats = preprocess_all_pdfs(
        pdf_dir=args.pdf_dir,
        use_vision=not args.no_vision,
        use_contextual=not args.no_contextual,
        use_raptor=not args.no_raptor,
        ocr_fallback=not args.no_ocr_fallback,
        clear_existing=args.clear,
    )

    print("\n=== RESULTS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    _cli()
