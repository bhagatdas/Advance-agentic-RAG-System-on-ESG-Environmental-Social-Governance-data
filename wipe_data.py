"""
wipe_data.py — Clear all preprocessed / runtime data while preserving the
source PDFs and the cross-encoder model cache.

Removes (then recreates as empty dirs so app paths stay valid):
  data/faiss/        — FAISS vector store
  data/tables/       — SQLite extracted tables
  data/checkpoints/  — LangGraph SqliteSaver conversation state
  data/images/       — extracted page images
  data/logs/         — application logs
  data/traces/       — local JSONL tracer output
  data/response_log.jsonl — API response log

Preserves:
  data/pdfs/   — source PDFs (re-ingestion would re-download these otherwise)
  data/models/ — cached cross-encoder weights (~90MB; re-download is slow)

Usage:
  python wipe_data.py              # interactive — prompts before deleting
  python wipe_data.py --yes        # non-interactive (CI / scripts)
  python wipe_data.py --data-dir D # override the data directory location
"""

import argparse
import shutil
from pathlib import Path

WIPE_DIRS = ["faiss", "tables", "checkpoints", "images", "logs", "traces"]
WIPE_FILES = ["response_log.jsonl"]
PRESERVE = ["pdfs", "models"]


def _count_files(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Wipe preprocessed data; preserve PDFs + model cache.")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--data-dir", default="data", help="Path to data directory (default: ./data).")
    args = parser.parse_args()

    data = Path(args.data_dir).resolve()
    if not data.exists():
        print(f"No data directory at {data} — nothing to wipe.")
        return

    target_dirs = [data / d for d in WIPE_DIRS if (data / d).exists()]
    target_files = [data / f for f in WIPE_FILES if (data / f).exists()]

    if not target_dirs and not target_files:
        print(f"Nothing to clean under {data}.")
        return

    print(f"\nAbout to remove from {data}:")
    for p in target_dirs:
        print(f"  {p.name}/  ({_count_files(p)} files)")
    for p in target_files:
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")
    print(f"\nPreserved: {', '.join(PRESERVE)}")

    if not args.yes:
        ans = input("\nContinue? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return

    print()
    for p in target_dirs:
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)
        print(f"  cleared {p.relative_to(data.parent)}/")
    for p in target_files:
        p.unlink(missing_ok=True)
        print(f"  removed {p.relative_to(data.parent)}")

    print(
        "\nDone. Re-ingest with:\n"
        "  python preprocessing.py --clear              # full pipeline\n"
        "  python preprocessing.py --clear --no-images  # skip image extraction (fastest)"
    )


if __name__ == "__main__":
    main()
