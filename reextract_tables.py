"""
Re-extract tables only — drops every table in data/tables.db, re-scans each PDF
in data/pdfs/ for tables, repopulates SQLite, and regenerates the schema catalog.

Does NOT touch FAISS, BM25, RAPTOR, or text/image chunks. Use this when the
SQL-side table extraction was wrong but the vector index is fine.

Usage:
    python reextract_tables.py
"""

import logging
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows so unicode chars in table descriptions don't crash
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config.settings import settings
from storage.sql_store import sql_store
from storage.schema_manager import schema_manager
from preprocessing import extract_tables_from_pdf
from utils.logging_config import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main() -> None:
    pdf_dir = Path(settings.pdf_dir)
    if not pdf_dir.exists():
        print(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(
        {p.resolve() for p in (*pdf_dir.glob("*.pdf"), *pdf_dir.glob("*.PDF"))}
    )
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) to re-scan")
    print("Dropping existing SQLite tables...")
    sql_store.drop_all_tables()

    total = 0
    for pdf in pdf_files:
        print(f"\nScanning: {pdf.name}")
        tables = extract_tables_from_pdf(str(pdf))
        total += len(tables)
        print(f"  -> {len(tables)} table(s) extracted")

    # Refresh schema catalog so the Table Agent sees the new shapes
    print("\nRegenerating schema catalog...")
    schema_manager.invalidate_cache()
    catalog = schema_manager.generate_catalog()

    print("\n=== DONE ===")
    print(f"Total tables in SQLite: {total}")
    print(f"Schema catalog size: {len(catalog)} chars")
    print("\nNote: FAISS still has the OLD table text representations.")
    print("If you want those refreshed too, run `python preprocessing.py --clear ...`")


if __name__ == "__main__":
    main()
