"""
Quick inspector for data/tables.db — prints every table's schema, row count,
and a sample of rows. Run with no args, or pass a table name to dump it fully.

Usage:
    python inspect_tables.py                  # list all tables + sample
    python inspect_tables.py <table_name>     # dump one table fully
    python inspect_tables.py --sql "SELECT ..."   # run an ad-hoc query
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Force UTF-8 on Windows so unicode chars (•, narrow no-break space, etc.) print fine
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DB = Path(__file__).parent / "data" / "tables.db"

if not DB.exists():
    print(f"DB not found at {DB}")
    sys.exit(1)


def list_all() -> None:
    conn = sqlite3.connect(DB)
    try:
        meta = pd.read_sql_query(
            "SELECT table_name, source_document, source_page, row_count, description "
            "FROM _table_metadata ORDER BY source_document, source_page",
            conn,
        )
        if meta.empty:
            print("No tables registered in _table_metadata.")
            return

        pd.set_option("display.max_colwidth", 60)
        pd.set_option("display.width", 200)
        print("\n=== Registered tables ===")
        print(meta.to_string(index=False))

        for tname in meta["table_name"]:
            print(f"\n--- {tname} (first 5 rows) ---")
            try:
                df = pd.read_sql_query(f"SELECT * FROM {tname} LIMIT 5", conn)
                print(df.to_string(index=False))
            except Exception as e:
                print(f"  [error reading {tname}: {e}]")
    finally:
        conn.close()


def dump_one(table: str) -> None:
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 250)
        print(f"\n=== {table} ({len(df)} rows) ===")
        print(df.to_string(index=False))
        out = Path(__file__).parent / "data" / f"{table}.csv"
        df.to_csv(out, index=False)
        print(f"\nWrote {out}")
    finally:
        conn.close()


def run_sql(sql: str) -> None:
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(sql, conn)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.width", 250)
        print(df.to_string(index=False))
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        list_all()
    elif sys.argv[1] == "--sql":
        run_sql(sys.argv[2])
    else:
        dump_one(sys.argv[1])
