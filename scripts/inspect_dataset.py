#!/usr/bin/env python3
"""
scripts/inspect_dataset.py

BingeBuddy - Phase 1 Dataset Inspection Script
==============================================
Safely inspects a TMDB TV Shows CSV dataset placed in data/raw/ without
modifying, overwriting, or deleting the file.

Designed specifically for systems with limited memory (4 GB RAM):
- Reads headers first.
- Processes large CSVs in conservative chunks (chunksize=10,000) to compute
  total row counts, missing-value counts, and duplicate counts.
- Avoids loading full text fields for 150,000+ rows into memory all at once.
- Strictly read-only: does not modify or save any files.
- If no CSV is present, reports clearly and exits without fabricating results.

Licensing note:
- Dataset Source: https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows
- License Status: UNVERIFIED (Pending review of Kaggle dataset page and applicable TMDB terms).
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

# Metadata constants
DATASET_SOURCE_URL = (
    "https://www.kaggle.com/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows"
)
LICENSE_STATUS = (
    "UNVERIFIED (Pending review of Kaggle dataset page and applicable TMDB terms)"
)

# Workspace directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Memory-safe chunk size for 4 GB RAM
CHUNK_SIZE = 10000


def find_csv_file(target_filename: Optional[str] = None) -> Optional[Path]:
    """
    Locates a CSV file in data/raw/.
    If target_filename is provided, searches for that specific file.
    Otherwise, picks the first CSV file found in data/raw/.
    """
    if not RAW_DATA_DIR.exists():
        return None

    if target_filename:
        specific_path = RAW_DATA_DIR / target_filename
        return specific_path if specific_path.is_file() else None

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    return csv_files[0] if csv_files else None


def format_bytes(size_bytes: int) -> str:
    """Formats file size into human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def inspect_dataset(csv_path: Path):
    """Performs memory-conservative inspection of the target CSV."""
    file_size = csv_path.stat().st_size

    print("\n" + "=" * 70)
    print(" BINGEBUDDY - DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f" Target File       : {csv_path.name}")
    print(f" Absolute Path     : {csv_path}")
    print(f" File Size         : {format_bytes(file_size)}")
    print(f" Source URL        : {DATASET_SOURCE_URL}")
    print(f" License Status    : {LICENSE_STATUS}")
    print("=" * 70)

    # 1. Read header only (nrows=0) to obtain actual column names safely
    try:
        header_df = pd.read_csv(csv_path, nrows=0)
        columns: List[str] = list(header_df.columns)
    except Exception as e:
        print(f"\n[!] Error reading CSV header: {e}")
        return

    print(f"\n[1] ACTUAL COLUMN NAMES ({len(columns)} columns detected)")
    print("-" * 70)
    for idx, col in enumerate(columns, start=1):
        print(f"  {idx:2d}. {col}")

    # 2. Inferred data types using a small sample (nrows=100)
    try:
        sample_types_df = pd.read_csv(csv_path, nrows=100)
        inferred_types = sample_types_df.dtypes.to_dict()
    except Exception as e:
        print(f"\n[!] Error reading sample rows for type inference: {e}")
        inferred_types = {}

    print("\n[2] INFERRED DATA TYPES (from initial sample)")
    print("-" * 70)
    for col in columns:
        dtype_str = str(inferred_types.get(col, "unknown"))
        print(f"  - {col:<30}: {dtype_str}")

    # 3. Chunked pass for row counts, missing values, and duplicate checks
    print(f"\n[3] SCANNING DATASET IN CHUNKS (chunksize={CHUNK_SIZE:,} rows)...")
    print("-" * 70)

    total_rows = 0
    missing_counts = {col: 0 for col in columns}
    seen_ids = set()
    duplicate_id_count = 0
    duplicate_row_hashes = 0
    seen_row_hashes = set()

    # Identify primary ID column if present (case-insensitive)
    id_col = None
    for col in columns:
        if col.lower() in ("id", "tmdb_id", "show_id"):
            id_col = col
            break

    try:
        chunk_iter = pd.read_csv(
            csv_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk_idx, chunk in enumerate(chunk_iter, start=1):
            chunk_len = len(chunk)
            total_rows += chunk_len

            # Missing values per column
            chunk_nulls = chunk.isnull().sum()
            for col in columns:
                missing_counts[col] += int(chunk_nulls[col])

            # Duplicate ID check (memory-efficient set lookup)
            if id_col is not None:
                for val in chunk[id_col].dropna():
                    if val in seen_ids:
                        duplicate_id_count += 1
                    else:
                        seen_ids.add(val)

            # Duplicate full-row check via 64-bit integer hash to preserve RAM
            # (150k integer hashes require < 2 MB RAM)
            for row_hash in pd.util.hash_pandas_object(chunk, index=False):
                if row_hash in seen_row_hashes:
                    duplicate_row_hashes += 1
                else:
                    seen_row_hashes.add(row_hash)

            print(f"    Processed chunk {chunk_idx:3d} ({total_rows:,} rows scanned)...", end="\r")

        print(f"\n    Completed scan: {total_rows:,} total rows processed.")
    except Exception as e:
        print(f"\n[!] Error during chunked scan: {e}")
        return

    print("\n[4] DATASET VOLUME & DUPLICATES")
    print("-" * 70)
    print(f"  Total records (rows)      : {total_rows:,}")
    if id_col:
        print(f"  Primary key column        : '{id_col}'")
        print(f"  Duplicate IDs found       : {duplicate_id_count:,}")
    else:
        print("  Primary key column        : None identified (no 'id' column found)")
    print(f"  Exact duplicate rows      : {duplicate_row_hashes:,}")

    print("\n[5] MISSING-VALUE SUMMARY")
    print("-" * 70)
    print(f"  {'Column Name':<30} | {'Missing Count':>14} | {'Missing %':>10}")
    print("  " + "-" * 60)
    for col in columns:
        count = missing_counts[col]
        pct = (count / total_rows * 100) if total_rows > 0 else 0.0
        flag = " [!] HIGH" if pct > 40.0 else ""
        print(f"  {col:<30} | {count:>14,} | {pct:>9.2f}%{flag}")

    # 4. Genre-related field analysis
    genre_cols = [c for c in columns if "genre" in c.lower()]
    print("\n[6] GENRE-RELATED FIELDS")
    print("-" * 70)
    if genre_cols:
        for g_col in genre_cols:
            print(f"  Field: '{g_col}'")
            # Sample first non-null values
            sample_genres = sample_types_df[g_col].dropna().head(3).tolist()
            print(f"    Sample values: {sample_genres}")
    else:
        print("  No genre-related columns detected.")

    # 5. Episode- and Season-related field analysis
    episode_season_cols = [
        c for c in columns if any(kw in c.lower() for kw in ("episode", "season"))
    ]
    print("\n[7] EPISODE & SEASON-RELATED FIELDS")
    print("-" * 70)
    if episode_season_cols:
        for es_col in episode_season_cols:
            sample_val = sample_types_df[es_col].dropna().head(2).tolist()
            print(f"  Field: '{es_col}' (type: {inferred_types.get(es_col, 'unknown')})")
            print(f"    Sample values: {sample_val}")

        # Assess whether data is episode-level or aggregate show-level
        is_aggregate = all(
            any(k in c.lower() for k in ("number_of", "count", "season", "episode", "run_time"))
            for c in episode_season_cols
        )
        if is_aggregate:
            print("\n  ASSESSMENT:")
            print("  * Detected fields are SHOW-LEVEL AGGREGATE METRICS")
            print("    (e.g., number_of_seasons, number_of_episodes, episode_run_time).")
            print("  * Individual episode-level records (episode titles, per-episode ratings,")
            print("    episode synopses, season-by-season breakdowns) are NOT present in this CSV.")
            print("  * In accordance with BingeBuddy requirements, episode details will")
            print("    NOT be fabricated or assumed.")
    else:
        print("  No episode or season fields detected.")

    # 6. Sample records (first 2 records, neatly formatted)
    print("\n[8] SAMPLE RECORDS (First 2 shows)")
    print("-" * 70)
    sample_preview = sample_types_df.head(2)
    for idx, (_, row) in enumerate(sample_preview.iterrows(), start=1):
        print(f"\n  --- Record #{idx} ---")
        for col in columns:
            val = str(row[col])
            # Truncate long text strings for readable terminal display
            if len(val) > 100:
                val = val[:97] + "..."
            print(f"    {col:<25}: {val}")

    print("\n" + "=" * 70)
    print(" END OF INSPECTION REPORT (No files were modified or deleted)")
    print("=" * 70 + "\n")


def main():
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = find_csv_file(target_arg)

    if not csv_path:
        print("\n" + "=" * 70)
        print(" BINGEBUDDY - DATASET INSPECTION")
        print("=" * 70)
        print(" [!] No CSV file found in 'data/raw/'.")
        print("     To inspect your dataset:")
        print("     1. Download the TMDB TV shows dataset from Kaggle:")
        print(f"        {DATASET_SOURCE_URL}")
        print("     2. Place the unzipped .csv file into:")
        print(f"        {RAW_DATA_DIR.resolve()}/")
        print("     3. Re-run this inspection script.\n")
        print(f" Dataset Source URL : {DATASET_SOURCE_URL}")
        print(f" Licensing Status   : {LICENSE_STATUS}")
        print("=" * 70 + "\n")
        sys.exit(0)

    inspect_dataset(csv_path)


if __name__ == "__main__":
    main()

