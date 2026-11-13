#!/usr/bin/env python3
"""
scripts/preprocess.py

BingeBuddy - Phase 2 Dataset Preprocessing Pipeline
===================================================
Cleans and optimizes the raw TMDB TV Shows CSV for the discovery & recommendation engine.

Features:
- Reads ONLY required columns confirmed by inspection.
- Strictly read-only on raw dataset: NEVER modifies, overwrites, or deletes raw CSV.
- Memory-conservative (< 50 MB RAM): type downcasting and efficient Pandas operations.
- Drops exact duplicate rows.
- Deterministic conflict resolution for duplicate IDs:
  Selection Rule:
    1. Metadata Completeness: Prioritizes records with non-empty overview, genres, and poster_path.
    2. Vote Count: Prioritizes records with higher vote_count (more reviewed/authoritative).
    3. Popularity: Prioritizes records with higher popularity score.
    4. Deterministic Tie-breaker: Retains first occurrence in original CSV.
- Excludes records with missing or blank titles ('name').
- Constructs clean 'recommendation_text' from available overview and genres without inventing content.
- Preserves valid numeric ratings, vote counts, popularity, dates, season/episode counts, and poster paths.
- Outputs clean CSV to 'data/processed/shows.csv'.
- Prints a comprehensive preprocessing report.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Workspace paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_CSV_PATH = PROCESSED_DATA_DIR / "shows.csv"

# Required columns confirmed by Phase 1 inspection
REQUIRED_COLUMNS = [
    "id",
    "name",
    "overview",
    "genres",
    "vote_average",
    "vote_count",
    "popularity",
    "first_air_date",
    "poster_path",
    "number_of_seasons",
    "number_of_episodes",
]

# Memory-efficient dtype specifications
DTYPE_SPECS = {
    "id": "int64",
    "name": "str",
    "overview": "str",
    "genres": "str",
    "vote_average": "float32",
    "vote_count": "int32",
    "popularity": "float32",
    "first_air_date": "str",
    "poster_path": "str",
    "number_of_seasons": "int16",
    "number_of_episodes": "int32",
}


def find_raw_csv() -> Path:
    """Locates the raw CSV dataset in data/raw/."""
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {RAW_DATA_DIR}.")
    return csv_files[0]


def construct_recommendation_text(overview: Any, genres: Any) -> str:
    """
    Constructs recommendation text combining genres and overview cleanly.
    Never invents or fabricates synthetic descriptions.
    """
    o_str = str(overview).strip() if pd.notnull(overview) else ""
    g_str = str(genres).strip() if pd.notnull(genres) else ""

    # Handle pandas 'nan' string artifact
    if o_str.lower() == "nan":
        o_str = ""
    if g_str.lower() == "nan":
        g_str = ""

    if g_str and o_str:
        return f"{g_str}. {o_str}"
    elif g_str:
        return g_str
    elif o_str:
        return o_str
    return ""


def resolve_conflicting_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Deterministically resolves duplicate IDs.
    
    Deterministic Selection Rule:
      1. Completeness Score (0-3): +1 each for non-empty overview, genres, and poster_path.
      2. Vote Count (descending): Higher vote count indicates more authoritative user reviews.
      3. Popularity (descending): Higher popularity metric.
      4. Original Position (ascending): Stable tiebreaker based on initial file order.
    
    Returns:
      (deduplicated_df, total_conflicting_rows, unique_conflicting_ids)
    """
    duplicate_mask = df.duplicated(subset=["id"], keep=False)
    total_conflicting_rows = int(duplicate_mask.sum())
    unique_conflicting_ids = int(df.loc[duplicate_mask, "id"].nunique())

    if total_conflicting_rows == 0:
        return df, 0, 0

    # Calculate completeness score without modifying original strings
    has_overview = (
        df["overview"].astype(str).str.strip().ne("")
        & df["overview"].astype(str).str.lower().ne("nan")
        & df["overview"].notnull()
    ).astype(int)

    has_genres = (
        df["genres"].astype(str).str.strip().ne("")
        & df["genres"].astype(str).str.lower().ne("nan")
        & df["genres"].notnull()
    ).astype(int)

    has_poster = (
        df["poster_path"].astype(str).str.strip().ne("")
        & df["poster_path"].astype(str).str.lower().ne("nan")
        & df["poster_path"].notnull()
    ).astype(int)

    df["_completeness"] = has_overview + has_genres + has_poster
    df["_orig_idx"] = df.index

    # Deterministic sort
    df_sorted = df.sort_values(
        by=["id", "_completeness", "vote_count", "popularity", "_orig_idx"],
        ascending=[True, False, False, False, True],
    )

    # Keep top-ranked record for each ID
    df_resolved = df_sorted.drop_duplicates(subset=["id"], keep="first").copy()

    # Clean up temporary columns
    df_resolved = df_resolved.drop(columns=["_completeness", "_orig_idx"])
    df = df.drop(columns=["_completeness", "_orig_idx"])

    return df_resolved, total_conflicting_rows, unique_conflicting_ids


def clean_dataset(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw DataFrame and returns (cleaned_df, report_metrics).
    Does not modify raw_df in place.
    """
    metrics: Dict[str, Any] = {}
    raw_rows = len(raw_df)
    metrics["raw_row_count"] = raw_rows

    df = raw_df.copy()

    # 1. Remove exact duplicate rows across all required columns
    exact_duplicates_count = int(df.duplicated().sum())
    metrics["exact_duplicates_removed"] = exact_duplicates_count
    if exact_duplicates_count > 0:
        df = df.drop_duplicates()

    # 2. Resolve conflicting duplicate IDs deterministically
    df, conflicting_rows, unique_conflicting_ids = resolve_conflicting_ids(df)
    metrics["conflicting_duplicate_rows"] = conflicting_rows
    metrics["unique_conflicting_ids"] = unique_conflicting_ids

    # 3. Exclude records with missing or blank titles ('name')
    missing_title_mask = (
        df["name"].isnull()
        | df["name"].astype(str).str.strip().eq("")
        | df["name"].astype(str).str.lower().eq("nan")
    )
    records_removed_missing_title = int(missing_title_mask.sum())
    metrics["records_removed_missing_title"] = records_removed_missing_title
    df = df[~missing_title_mask].copy()

    # 4. Handle text fields safely (no synthetic content, fill na with empty string)
    for col in ["overview", "genres", "poster_path", "first_air_date"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
        df.loc[df[col].str.lower() == "nan", col] = ""

    # Ensure names are trimmed strings
    df["name"] = df["name"].astype(str).str.strip()

    # 5. Handle numeric fields safely (handle NaN, invalid values, clamp ranges)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_average"] = df["vote_average"].clip(0.0, 10.0).astype(np.float32)

    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df["vote_count"] = df["vote_count"].clip(lower=0).astype(np.int32)

    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    df["popularity"] = df["popularity"].clip(lower=0.0).astype(np.float32)

    df["number_of_seasons"] = pd.to_numeric(df["number_of_seasons"], errors="coerce").fillna(0)
    df["number_of_seasons"] = df["number_of_seasons"].clip(lower=0).astype(np.int16)

    df["number_of_episodes"] = pd.to_numeric(df["number_of_episodes"], errors="coerce").fillna(0)
    df["number_of_episodes"] = df["number_of_episodes"].clip(lower=0).astype(np.int32)

    # 6. Construct recommendation_text without inventing metadata
    df["recommendation_text"] = [
        construct_recommendation_text(o, g)
        for o, g in zip(df["overview"], df["genres"])
    ]

    # 7. Sort by popularity descending for fast UI discovery
    df = df.sort_values(by="popularity", ascending=False).reset_index(drop=True)

    # Metrics calculation
    final_rows = len(df)
    unique_ids = int(df["id"].nunique())
    metrics["final_row_count"] = final_rows
    metrics["unique_id_count"] = unique_ids
    metrics["missing_overview_count"] = int(df["overview"].eq("").sum())
    metrics["missing_genres_count"] = int(df["genres"].eq("").sum())
    metrics["missing_poster_count"] = int(df["poster_path"].eq("").sum())
    metrics["empty_recommendation_text_count"] = int(df["recommendation_text"].eq("").sum())

    # Guarantee unique IDs
    assert final_rows == unique_ids, (
        f"ID uniqueness check failed: {final_rows} rows vs {unique_ids} unique IDs"
    )

    return df, metrics


def format_bytes(size_bytes: int) -> str:
    """Formats file size into human-readable units."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def main():
    try:
        raw_csv_path = find_raw_csv()
    except FileNotFoundError as err:
        print(f"[!] {err}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" BINGEBUDDY - DATASET PREPROCESSING (PHASE 2)")
    print("=" * 70)
    print(f" Raw dataset source   : {raw_csv_path.name}")
    print(f" Raw file size        : {format_bytes(raw_csv_path.stat().st_size)}")
    print(f" Destination output   : {OUTPUT_CSV_PATH}")
    print("-" * 70)
    print(" [*] Loading required columns safely...")

    # Load ONLY required columns with optimized types to respect 4 GB RAM
    raw_df = pd.read_csv(
        raw_csv_path,
        usecols=REQUIRED_COLUMNS,
        dtype=DTYPE_SPECS,
        low_memory=False,
    )

    print(f" [*] Cleaning and deduplicating {len(raw_df):,} records...")
    cleaned_df, metrics = clean_dataset(raw_df)

    # Ensure destination directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to shows.csv
    print(f" [*] Writing processed dataset to {OUTPUT_CSV_PATH.name}...")
    cleaned_df.to_csv(OUTPUT_CSV_PATH, index=False)

    out_file_size = OUTPUT_CSV_PATH.stat().st_size

    # Print Preprocessing Report
    print("\n" + "=" * 70)
    print(" PREPROCESSING COMPLETION REPORT")
    print("=" * 70)
    print(f" Raw row count                      : {metrics['raw_row_count']:,}")
    print(f" Exact duplicate rows removed       : {metrics['exact_duplicates_removed']:,}")
    print(f" Conflicting duplicate rows found   : {metrics['conflicting_duplicate_rows']:,}")
    print(f" Unique conflicting IDs resolved    : {metrics['unique_conflicting_ids']:,}")
    print(f" Records removed (missing title)    : {metrics['records_removed_missing_title']:,}")
    print(f" Final row count                    : {metrics['final_row_count']:,}")
    print(f" Unique ID count (guaranteed unique): {metrics['unique_id_count']:,}")
    print("-" * 70)
    print(" Metadata & Feature Completeness:")
    print(f"  - Missing overview count          : {metrics['missing_overview_count']:,} ({(metrics['missing_overview_count'] / metrics['final_row_count'] * 100):.2f}%)")
    print(f"  - Missing genres count            : {metrics['missing_genres_count']:,} ({(metrics['missing_genres_count'] / metrics['final_row_count'] * 100):.2f}%)")
    print(f"  - Missing poster_path count       : {metrics['missing_poster_count']:,} ({(metrics['missing_poster_count'] / metrics['final_row_count'] * 100):.2f}%)")
    print(f"  - Empty recommendation text count : {metrics['empty_recommendation_text_count']:,} ({(metrics['empty_recommendation_text_count'] / metrics['final_row_count'] * 100):.2f}%)")
    print("-" * 70)
    print(f" Output file size                   : {format_bytes(out_file_size)}")
    print(f" Status                             : SUCCESS - Ready for Phase 3 (API & Loader)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
